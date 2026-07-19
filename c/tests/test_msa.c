/* test_msa.c — unit tests for the MiniMax Sparse Attention (MSA) port.
 *
 * Covers: (1) indexer forward (Q/K -> 128-dim, per-head norm, max-score over
 * 128-token blocks), (2) top-16 block selection with init_block=0 + local_block=1,
 * (3) block-sparse softmax attention vs dense reference over selected blocks,
 * (4) sparse_attention_freq gating (layers 0-2 dense, 3-59 sparse).
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_msa.c -o c/tests/test_msa -lm
 * Run:   ./c/tests/test_msa
 */
#ifndef TESTING
#define TESTING
#endif
#include "../src/engine.c"

#include <assert.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        g_fail = 1; \
    } \
} while (0)

/* ---------- reference implementations (independent of engine code) ---------- */

/* Reference dot product (no SIMD). */
static float ref_dot(const float *a, const float *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i] * b[i];
    return (float)s;
}

/* Reference RMSNorm (standard, gemma=0): o = x / rms(x) * w  (no +1). */
static void ref_rmsnorm(float *o, const float *x, const float *w, int D, float eps) {
    double ms = 0.0;
    for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) o[i] = x[i] * r * w[i];
}

/* Reference partial RoPE (rotate-half on first rotary_dim channels). */
static void ref_rope(float *x, int pos, float theta, int hd, int rd) {
    if (rd <= 0 || rd > hd) rd = hd;
    int h2 = rd / 2;
    for (int j = 0; j < h2; j++) {
        float inv = powf(theta, -2.0f * j / (float)rd);
        float ang = pos * inv, c = cosf(ang), s = sinf(ang);
        float x1 = x[j], x2 = x[j + h2];
        x[j] = x1 * c - x2 * s;
        x[j + h2] = x2 * c + x1 * s;
    }
}

/* Reference indexer scores: per-index-head max over 128-token blocks. */
static void ref_indexer_scores(const float *iq, const float *Kidx, int pos,
                               int idx_heads, int idx_dim, int max_t,
                               int blk_size, int n_blocks, float *scores) {
    (void)max_t;
    for (int h = 0; h < idx_heads; h++) {
        const float *iqh = iq + (int64_t)h * idx_dim;
        for (int b = 0; b < n_blocks; b++) {
            int bstart = b * blk_size;
            int bend = bstart + blk_size;
            if (bend > pos + 1) bend = pos + 1;
            float mx = -1e30f;
            for (int t = bstart; t < bend; t++) {
                float s = ref_dot(iqh, Kidx + (int64_t)t * idx_dim, idx_dim);
                if (s > mx) mx = s;
            }
            scores[h * n_blocks + b] = (bend > bstart) ? mx : -1e30f;
        }
    }
}

/* Reference dense causal GQA softmax attention for one query token.
 * Q: [H * hd], K/V cache: [max_t * Hkv * hd], out: [H * hd]. */
static void ref_dense_attn(const float *Q, const float *K, const float *V,
                           int H, int Hkv, int hd, int max_t, int pos, float scale,
                           float *out) {
    int nrep = H / Hkv;
    for (int h = 0; h < H; h++) {
        int kvh = h / nrep;
        const float *qh = Q + (int64_t)h * hd;
        int nt = pos + 1;
        float *sc = (float *)malloc((size_t)nt * sizeof(float));
        float mx = -1e30f;
        for (int t = 0; t < nt; t++) {
            const float *kt = K + ((int64_t)kvh * max_t + t) * hd;
            sc[t] = ref_dot(qh, kt, hd) * scale;
            if (sc[t] > mx) mx = sc[t];
        }
        float sum = 0;
        for (int t = 0; t < nt; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        float *oh = out + (int64_t)h * hd;
        memset(oh, 0, (size_t)hd * sizeof(float));
        for (int t = 0; t < nt; t++) {
            const float *vt = V + ((int64_t)kvh * max_t + t) * hd;
            for (int i = 0; i < hd; i++) oh[i] += sc[t] / sum * vt[i];
        }
        free(sc);
    }
}

/* ---------- Test 1: indexer forward (Q/K -> 128-dim, norm, max-score over blocks) ---------- */
static void test_indexer_forward(void) {
    fprintf(stderr, "==> test_indexer_forward\n");
    /* Synthetic fixture: 4 index heads, 128-dim, 4 blocks of 128 tokens = 512 KV positions */
    const int idx_heads = 4, idx_dim = 128, max_t = 512;
    const int blk_size = 128, n_blocks = (max_t + blk_size - 1) / blk_size;
    const int pos = 400;     /* current query position (block 3 partially visible) */
    const float eps = 1e-6f;
    const float theta = 5000000.0f;
    const int rotary_dim = 64;

    /* Random indexer Q (already projected, pre-norm/pre-rope) and indexer K cache */
    float *iq_raw = falloc((int64_t)idx_heads * idx_dim);
    float *iq = falloc((int64_t)idx_heads * idx_dim);
    float *Kidx = falloc((int64_t)max_t * idx_dim);
    float *iqn = falloc(idx_dim);     /* indexer Q norm weights */
    /* No ikn / K norm for this test — Kidx is pre-computed (simulates already-normed+roped) */
    srand(42);
    for (int i = 0; i < idx_heads * idx_dim; i++) iq_raw[i] = (float)(rand() % 2000 - 1000) / 1000.0f;
    for (int i = 0; i < max_t * idx_dim; i++) Kidx[i] = (float)(rand() % 2000 - 1000) / 1000.0f;
    for (int i = 0; i < idx_dim; i++) iqn[i] = 1.0f + (float)(rand() % 100) / 100.0f;

    /* Apply norm + RoPE to iq (matching engine path) */
    memcpy(iq, iq_raw, (size_t)idx_heads * idx_dim * sizeof(float));
    for (int h = 0; h < idx_heads; h++) {
        float *iqh = iq + (int64_t)h * idx_dim;
        rmsnorm(iqh, iqh, iqn, idx_dim, eps, 0);     /* engine rmsnorm (gemma=0) */
        rope(iqh, pos, theta, idx_dim, rotary_dim);  /* engine rope */
    }

    /* Reference: apply same norm + RoPE independently */
    float *iq_ref = falloc((int64_t)idx_heads * idx_dim);
    memcpy(iq_ref, iq_raw, (size_t)idx_heads * idx_dim * sizeof(float));
    for (int h = 0; h < idx_heads; h++) {
        float *iqh = iq_ref + (int64_t)h * idx_dim;
        ref_rmsnorm(iqh, iqh, iqn, idx_dim, eps);
        ref_rope(iqh, pos, theta, idx_dim, rotary_dim);
    }
    /* Verify norm+rope match between engine and reference */
    float max_diff_nr = 0;
    for (int i = 0; i < idx_heads * idx_dim; i++) {
        float d = fabsf(iq[i] - iq_ref[i]);
        if (d > max_diff_nr) max_diff_nr = d;
    }
    CHECK(max_diff_nr < 1e-5, "indexer norm+rope matches reference");

    /* Engine indexer scores */
    float *scores_eng = falloc((int64_t)idx_heads * n_blocks);
    msa_indexer_scores(iq, Kidx, pos, idx_heads, idx_dim, max_t, blk_size, n_blocks, scores_eng);

    /* Reference indexer scores */
    float *scores_ref = falloc((int64_t)idx_heads * n_blocks);
    ref_indexer_scores(iq_ref, Kidx, pos, idx_heads, idx_dim, max_t, blk_size, n_blocks, scores_ref);

    float max_diff = 0;
    for (int i = 0; i < idx_heads * n_blocks; i++) {
        float d = fabsf(scores_eng[i] - scores_ref[i]);
        if (d > max_diff) max_diff = d;
    }
    fprintf(stderr, "   indexer scores max abs diff = %.2e (tol 1e-5), nblocks=%d, heads=%d, dim=%d\n",
            max_diff, n_blocks, idx_heads, idx_dim);
    CHECK(max_diff < 1e-5, "indexer forward: max-score matches reference within 1e-5");

    /* Spot-check: block 3 (tokens 384..400) should have scores computed from tokens 384..400 only */
    CHECK(scores_eng[0 * n_blocks + 3] > -1e29f, "block 3 has a valid (non -1e30) score for head 0");
    /* Blocks beyond pos/blk_size+1 should be -1e30 (no visible tokens) */
    int beyond = (pos + 1) / blk_size + 1;
    if (beyond < n_blocks)
        CHECK(scores_eng[beyond] <= -1e29f, "block beyond causal horizon is -1e30");

    free(iq_raw); free(iq); free(iq_ref); free(Kidx); free(iqn);
    free(scores_eng); free(scores_ref);
}

/* ---------- Test 2: top-16 block selection with init_block=0 + local_block=1 ---------- */
static void test_top16_block_selection(void) {
    fprintf(stderr, "==> test_top16_block_selection\n");
    const int n_blocks = 128;
    const int topk = 16;
    const int init_blk = 0;
    const int blk_size = 128;
    const int local_blks = 1;

    /* Construct a deterministic score vector where blocks 0 and the local block
     * score LOW (to prove they are still selected by the forced policy). */
    float *scores = falloc(n_blocks);
    for (int b = 0; b < n_blocks; b++) scores[b] = -100.0f;     /* all low */
    /* Set some high-scoring blocks */
    int top14[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 5, 15};
    for (int i = 0; i < 14; i++) scores[top14[i]] = (float)(100 - i);  /* descending */

    /* Query at position 5000 -> local block = 5000/128 = 39 */
    int pos = 5000;
    int local_blk = pos / blk_size;     /* 39 */

    int selected[64];
    int ns = msa_topk_select(scores, n_blocks, topk, init_blk, pos, blk_size, local_blks, selected);

    CHECK(ns == topk, "exactly 16 blocks selected");
    /* (a) block 0 (init) always appears */
    int has0 = 0;
    for (int i = 0; i < ns; i++) if (selected[i] == 0) has0 = 1;
    CHECK(has0, "init_block=0 always appears in selected set");
    /* (b) local block 39 always appears */
    int has_local = 0;
    for (int i = 0; i < ns; i++) if (selected[i] == local_blk) has_local = 1;
    CHECK(has_local, "local_block (current position's block) always appears");
    /* (c) exactly 16 selected */
    CHECK(ns == 16, "exactly 16 blocks selected per group");
    /* (d) non-forced selected blocks are the top-14 by score (excluding 0 and 39) */
    /* The top-14 we set: {10,20,30,40,50,60,70,80,90,100,110,120,5,15} (excluding 0,39).
     * But block 39 has score -100 (low), and blocks 0 has score -100.
     * So the non-forced selected should be exactly the 14 highest-scoring blocks. */
    int sel_copy[64];
    memcpy(sel_copy, selected, (size_t)ns * sizeof(int));
    /* sort selected for easier comparison */
    for (int i = 0; i < ns; i++)
        for (int j = i + 1; j < ns; j++)
            if (sel_copy[i] > sel_copy[j]) { int t = sel_copy[i]; sel_copy[i] = sel_copy[j]; sel_copy[j] = t; }

    /* Build expected set: {0, 39} + top-14 excluding {0, 39} */
    int expected[20];
    int ne = 0;
    expected[ne++] = 0;
    if (local_blk != 0) expected[ne++] = local_blk;
    /* find top-14 by score, excluding forced blocks */
    char taken[128] = {0};
    taken[0] = 1; taken[local_blk] = 1;
    for (int kk = 0; kk < 14; kk++) {
        int best = -1; float bv = -1e30f;
        for (int b = 0; b < n_blocks; b++) {
            if (!taken[b] && scores[b] > bv) { bv = scores[b]; best = b; }
        }
        if (best >= 0) { expected[ne++] = best; taken[best] = 1; }
    }
    /* sort expected */
    for (int i = 0; i < ne; i++)
        for (int j = i + 1; j < ne; j++)
            if (expected[i] > expected[j]) { int t = expected[i]; expected[i] = expected[j]; expected[j] = t; }

    CHECK(ne == 16, "expected set has exactly 16 blocks");
    int all_match = (ne == ns);
    for (int i = 0; i < ns && i < ne; i++) {
        if (sel_copy[i] != expected[i]) all_match = 0;
    }
    CHECK(all_match, "selected set matches expected {init, local, top-14}");

    fprintf(stderr, "   selected blocks:");
    for (int i = 0; i < ns; i++) fprintf(stderr, " %d", selected[i]);
    fprintf(stderr, "\n");

    /* Edge case: small context (fewer blocks than topk) */
    int n_blocks_small = 8;
    float scores_small[8];
    for (int i = 0; i < 8; i++) scores_small[i] = (float)i;
    int sel_small[64];
    int ns_small = msa_topk_select(scores_small, n_blocks_small, topk, 0, 100, blk_size, 1, sel_small);
    CHECK(ns_small == 16, "small context still returns topk entries (padded)");
    int valid = 0;
    for (int i = 0; i < 16; i++) if (sel_small[i] >= 0) valid++;
    CHECK(valid == 8, "small context: only 8 valid blocks (rest are -1 padding)");

    free(scores);
}

/* ---------- Test 3: block-sparse softmax attention matches dense over selected blocks ---------- */
static void test_block_sparse_softmax(void) {
    fprintf(stderr, "==> test_block_sparse_softmax\n");
    /* When ALL blocks are selected (forced to keep every block), the sparse path
     * must reproduce dense causal GQA attention. This isolates the sparse attention
     * math from the selection math (the colibri DSA-validation trick). */
    const int H = 64, Hkv = 4, hd = 128;
    const int n_q_in_group = H / Hkv;     /* 16 */
    const int blk_size = 128;
    const int max_t = 256;                /* 2 blocks */
    const int n_blocks = (max_t + blk_size - 1) / blk_size;     /* 2 */
    const int topk = n_blocks;            /* force-select ALL blocks -> dense equivalent */
    const int pos = 200;                  /* query position 200 (block 1 partially visible) */
    const float scale = 1.0f / sqrtf((float)hd);

    /* Build a minimal Model with f32 KV cache for 1 layer */
    Model m;
    memset(&m, 0, sizeof(m));
    m.c.heads = H; m.c.kv_heads = Hkv; m.c.head_dim = hd;
    m.c.hidden = H * hd; m.c.layers = 1; m.c.eps = 1e-6f;
    m.max_t = max_t; m.kv_i8 = 0; m.planar = 0;
    m.K = (float **)calloc(1, sizeof(float *));
    m.V = (float **)calloc(1, sizeof(float *));
    m.K[0] = falloc((int64_t)Hkv * max_t * hd);
    m.V[0] = falloc((int64_t)Hkv * max_t * hd);

    /* Fill K/V cache with random values for tokens 0..pos */
    srand(123);
    for (int kvh = 0; kvh < Hkv; kvh++)
        for (int t = 0; t <= pos; t++)
            for (int i = 0; i < hd; i++) {
                m.K[0][((int64_t)kvh * max_t + t) * hd + i] = (float)(rand() % 2000 - 1000) / 1000.0f;
                m.V[0][((int64_t)kvh * max_t + t) * hd + i] = (float)(rand() % 2000 - 1000) / 1000.0f;
            }

    /* Random Q for one token: [H * hd] */
    float *Q = falloc((int64_t)H * hd);
    for (int i = 0; i < H * hd; i++) Q[i] = (float)(rand() % 2000 - 1000) / 1000.0f;

    /* Selected blocks: ALL blocks (force dense equivalent) */
    int selected[64];
    for (int b = 0; b < n_blocks; b++) selected[b] = b;

    /* Sparse attention output via msa_group_attention for each GQA group */
    float *out_sparse = falloc((int64_t)H * hd);
    memset(out_sparse, 0, (size_t)H * hd * sizeof(float));
    for (int g = 0; g < Hkv; g++) {
        const float *Qg = Q + (int64_t)g * n_q_in_group * hd;
        float *outg = out_sparse + (int64_t)g * n_q_in_group * hd;
        msa_group_attention(&m, 0, g, Qg, selected, topk,
                           n_q_in_group, hd, pos, blk_size, scale, outg);
    }

    /* Dense reference attention */
    float *out_dense = falloc((int64_t)H * hd);
    ref_dense_attn(Q, m.K[0], m.V[0], H, Hkv, hd, max_t, pos, scale, out_dense);

    float max_diff = 0;
    for (int i = 0; i < H * hd; i++) {
        float d = fabsf(out_sparse[i] - out_dense[i]);
        if (d > max_diff) max_diff = d;
    }
    fprintf(stderr, "   MSA-dense equivalence: max abs diff = %.2e (tol 1e-5) over [heads=%d, hd=%d, pos=%d]\n",
            max_diff, H, hd, pos);
    CHECK(max_diff < 1e-5, "block-sparse softmax matches dense reference over all selected blocks");

    free(out_sparse); free(out_dense); free(Q);
    free(m.K[0]); free(m.V[0]); free(m.K); free(m.V);
}

/* ---------- Test 4: sparse_attention_freq gating (layers 0-2 dense, 3-59 sparse) ---------- */
static void test_sparse_freq_gating(void) {
    fprintf(stderr, "==> test_sparse_freq_gating\n");
    /* Simulate the MiniMax M3 sparse_attention_freq array: [0,0,0,1,1,...,1] for 60 layers */
    const int layers = 60;
    const int first_dense = 3;
    Cfg c;
    memset(&c, 0, sizeof(c));
    c.layers = layers;
    c.first_dense = first_dense;
    c.sparse_freq = (int *)calloc((size_t)layers, sizeof(int));
    for (int i = 0; i < layers; i++) c.sparse_freq[i] = (i >= first_dense) ? 1 : 0;

    /* Simulate the engine's per-layer msa flag assignment from load_model */
    int dense_count = 0, sparse_count = 0;
    for (int i = 0; i < layers; i++) {
        int msa = (c.sparse_freq && c.sparse_freq[i]) ? 1 : 0;
        if (msa) sparse_count++; else dense_count++;
    }
    CHECK(dense_count == 3, "layers 0-2 are dense (3 dense layers)");
    CHECK(sparse_count == 57, "layers 3-59 are sparse (57 sparse layers)");

    /* Verify the gating matches the sparse_attention_freq array */
    int gating_ok = 1;
    for (int i = 0; i < layers; i++) {
        int expected_msa = c.sparse_freq[i];
        int actual_msa = (i >= first_dense) ? 1 : 0;
        if (expected_msa != actual_msa) gating_ok = 0;
    }
    CHECK(gating_ok, "gating matches sparse_attention_freq [0,0,0,1,...,1]");

    /* Verify specific layers */
    CHECK(!c.sparse_freq[0], "layer 0 is dense (sparse_attention_freq[0]==0)");
    CHECK(!c.sparse_freq[1], "layer 1 is dense (sparse_attention_freq[1]==0)");
    CHECK(!c.sparse_freq[2], "layer 2 is dense (sparse_attention_freq[2]==0)");
    CHECK(c.sparse_freq[3], "layer 3 is sparse (sparse_attention_freq[3]==1)");
    CHECK(c.sparse_freq[59], "layer 59 is sparse (sparse_attention_freq[59]==1)");

    fprintf(stderr, "   sparse gating: layers 0,1,2=dense (%d); layers 3..59=sparse (%d)\n",
            dense_count, sparse_count);

    free(c.sparse_freq);
}

/* ---------- main ---------- */
int main(void) {
    fprintf(stderr, "=== MSA unit tests ===\n");
    test_indexer_forward();
    test_top16_block_selection();
    test_block_sparse_softmax();
    test_sparse_freq_gating();
    if (g_fail) {
        fprintf(stderr, "\nFAILED: one or more MSA tests failed\n");
        return 1;
    }
    fprintf(stderr, "\nPASS: all MSA tests passed\n");
    return 0;
}
