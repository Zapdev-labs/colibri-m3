/* eagle3.h - EAGLE3 draft model for speculative decoding
 *
 * Architecture (corrected from GGUF layout):
 *   fc: [18432] -> [6144]  (takes 3 concatenated hidden states, outputs 1)
 *   fc_norm.0/1/2: 3 x [6144] RMSNorm for the 3 input parts
 *   blk.0: 1 transformer layer
 *     - attn_norm: [6144] before attention
 *     - Q/K/V: [12288] -> [8192] (input = concat of 2 parts of 6144)
 *     - attn_output: [8192] -> [6144]
 *     - attn_norm_2: [6144] norm on attn output
 *     - ffn_norm: [6144] before FFN
 *     - gate/up: [6144] -> [18432]
 *     - down: [18432] -> [6144]
 *   output_norm + lm_head: [6144] -> [vocab]
 */
#ifndef COLIBRI_M3_EAGLE3_H
#define COLIBRI_M3_EAGLE3_H

#include <sys/mman.h>

#define E3_HIDDEN       6144
#define E3_N_HEADS      64
#define E3_HEAD_DIM     128
#define E3_KV_HEADS     64
#define E3_FFN_INTER    18432
#define E3_FC_IN        18432   /* 3 * 6144 */
#define E3_QKV_IN       12288   /* 2 * 6144 */
#define E3_QKV_OUT      8192    /* 64 * 128 */
#define E3_ROPE_DIM     128
#define E3_MAX_DRAFT    4
#define E3_TARGET_LAYER 57

typedef struct {
    float *fc_w;            /* [6144, 18432] = (out=6144, in=18432) */
    float *fc_norm[3];      /* 3 x [6144] */
    float *attn_norm;       /* [6144] */
    float *attn_norm_2;     /* [6144] */
    float *attn_q;          /* [8192, 12288] = (out=8192, in=12288) */
    float *attn_k;          /* [8192, 12288] */
    float *attn_v;          /* [8192, 12288] */
    float *attn_output;     /* [6144, 8192] = (out=6144, in=8192) */
    float *ffn_norm;        /* [6144] */
    float *ffn_gate;        /* [18432, 6144] = (out=18432, in=6144) */
    float *ffn_up;          /* [18432, 6144] */
    float *ffn_down;        /* [6144, 18432] = (out=6144, in=18432) */
    float *output_norm;     /* [6144] */
    float *output_w;         /* [6144, 200064] = (out=200064, in=6144) */
    float *K_cache;
    float *V_cache;
    int loaded;
    int max_pos;
    float *scratch_x;       /* [6144] */
    float *fc_in_buf;       /* [18432] */
    float *qkv_in_buf;      /* [12288] */
} Eagle3State;

static Eagle3State g_e3;

static float *e3_load_raw(const char *dir, const char *name, int n) {
    char path[1024];
    char safe[512];
    int j = 0;
    for (int i = 0; name[i] && j < 511; i++)
        safe[j++] = (name[i] == '.') ? '_' : name[i];
    safe[j] = 0;
    snprintf(path, sizeof(path), "%s/%s.f32", dir, safe);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[e3] missing: %s\n", path); return NULL; }
    float *p = falloc(n);
    fread(p, 1, (size_t)n * sizeof(float), f);
    fclose(f);
    return p;
}

static void e3_load(const char *eagle_dir) {
    if (!eagle_dir) eagle_dir = "/home/ai/models/eagle3_raw";
    fprintf(stderr, "[e3] loading from %s\n", eagle_dir);
    memset(&g_e3, 0, sizeof(g_e3));

    int D = E3_HIDDEN;
    g_e3.fc_w = e3_load_raw(eagle_dir, "fc.weight", D * E3_FC_IN);
    g_e3.fc_norm[0] = e3_load_raw(eagle_dir, "fc_norm.0.weight", D);
    g_e3.fc_norm[1] = e3_load_raw(eagle_dir, "fc_norm.1.weight", D);
    g_e3.fc_norm[2] = e3_load_raw(eagle_dir, "fc_norm.2.weight", D);
    g_e3.attn_norm = e3_load_raw(eagle_dir, "blk.0.attn_norm.weight", D);
    g_e3.attn_norm_2 = e3_load_raw(eagle_dir, "blk.0.attn_norm_2.weight", D);
    g_e3.attn_q = e3_load_raw(eagle_dir, "blk.0.attn_q.weight", E3_QKV_OUT * E3_QKV_IN);
    g_e3.attn_k = e3_load_raw(eagle_dir, "blk.0.attn_k.weight", E3_QKV_OUT * E3_QKV_IN);
    g_e3.attn_v = e3_load_raw(eagle_dir, "blk.0.attn_v.weight", E3_QKV_OUT * E3_QKV_IN);
    g_e3.attn_output = e3_load_raw(eagle_dir, "blk.0.attn_output.weight", D * E3_QKV_OUT);
    g_e3.ffn_norm = e3_load_raw(eagle_dir, "blk.0.ffn_norm.weight", D);
    g_e3.ffn_gate = e3_load_raw(eagle_dir, "blk.0.ffn_gate.weight", E3_FFN_INTER * D);
    g_e3.ffn_up = e3_load_raw(eagle_dir, "blk.0.ffn_up.weight", E3_FFN_INTER * D);
    g_e3.ffn_down = e3_load_raw(eagle_dir, "blk.0.ffn_down.weight", D * E3_FFN_INTER);
    g_e3.output_norm = e3_load_raw(eagle_dir, "output_norm.weight", D);
    g_e3.output_w = e3_load_raw(eagle_dir, "output.weight", E3_HIDDEN * 200064);

    g_e3.max_pos = 8192;
    g_e3.K_cache = falloc((int64_t)g_e3.max_pos * E3_KV_HEADS * E3_HEAD_DIM);
    g_e3.V_cache = falloc((int64_t)g_e3.max_pos * E3_KV_HEADS * E3_HEAD_DIM);
    memset(g_e3.K_cache, 0, (size_t)g_e3.max_pos * E3_KV_HEADS * E3_HEAD_DIM * sizeof(float));
    memset(g_e3.V_cache, 0, (size_t)g_e3.max_pos * E3_KV_HEADS * E3_HEAD_DIM * sizeof(float));

    g_e3.scratch_x = falloc(D);
    g_e3.fc_in_buf = falloc(E3_FC_IN);
    g_e3.qkv_in_buf = falloc(E3_QKV_IN);

    g_e3.loaded = 1;
    fprintf(stderr, "[e3] loaded: 1 layer, %d heads, head_dim=%d, ffn=%d\n",
            E3_N_HEADS, E3_HEAD_DIM, E3_FFN_INTER);
}

/* EAGLE3 attention (standard GQA) */
static void e3_attention(float *qkv_in, int pos, float *out, Cfg *c) {
    int nh = E3_N_HEADS, hd = E3_HEAD_DIM, nkv = E3_KV_HEADS;
    int qkv_dim = E3_QKV_OUT;
    float scale = 1.f / sqrtf((float)hd);

    float *q = falloc(qkv_dim), *k = falloc(qkv_dim), *v = falloc(qkv_dim);
    /* Q/K/V: (out=8192, in=12288) */
    matmul_f(q, qkv_in, g_e3.attn_q, 1, E3_QKV_IN, qkv_dim);
    matmul_f(k, qkv_in, g_e3.attn_k, 1, E3_QKV_IN, qkv_dim);
    matmul_f(v, qkv_in, g_e3.attn_v, 1, E3_QKV_IN, qkv_dim);

    for (int h = 0; h < nh; h++) rope(q + (int64_t)h * hd, pos, c->theta, hd, E3_ROPE_DIM);
    for (int h = 0; h < nkv; h++) rope(k + (int64_t)h * hd, pos, c->theta, hd, E3_ROPE_DIM);

    int64_t kv_off = (int64_t)pos * nkv * hd;
    memcpy(g_e3.K_cache + kv_off, k, (size_t)nkv * hd * sizeof(float));
    memcpy(g_e3.V_cache + kv_off, v, (size_t)nkv * hd * sizeof(float));

    int nt = pos + 1;
    float *ctx = falloc(qkv_dim);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < nh; h++) {
        int kvh = h * nkv / nh;
        const float *qh = q + (int64_t)h * hd;
        float *och = ctx + (int64_t)h * hd;
        float *sc = (float *)malloc((size_t)nt * sizeof(float));
        float mx = -1e30f;
        for (int t = 0; t < nt; t++) {
            const float *kh = g_e3.K_cache + (int64_t)t * nkv * hd + (int64_t)kvh * hd;
            float dot = 0;
            for (int i = 0; i < hd; i++) dot += qh[i] * kh[i];
            sc[t] = dot * scale;
            if (sc[t] > mx) mx = sc[t];
        }
        float sum = 0;
        for (int t = 0; t < nt; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        float inv = sum > 0 ? 1.f / sum : 0;
        memset(och, 0, (size_t)hd * sizeof(float));
        for (int t = 0; t < nt; t++) {
            float w = sc[t] * inv;
            const float *vh = g_e3.V_cache + (int64_t)t * nkv * hd + (int64_t)kvh * hd;
            for (int i = 0; i < hd; i++) och[i] += w * vh[i];
        }
        free(sc);
    }

    /* attn_output: (out=6144, in=8192) */
    matmul_f(out, ctx, g_e3.attn_output, 1, qkv_dim, E3_HIDDEN);
    free(q); free(k); free(v); free(ctx);
}

/* EAGLE3 forward: predict next token.
 * base_hidden: [6144] - hidden state from base model layer 57
 * token_embed: [6144] - embedding of the predicted token
 * pos: current position
 * logits: [vocab] - output logits
 * Returns: predicted token id */
static int e3_forward(float *base_hidden, float *token_embed, int pos,
                       float *logits, Cfg *c, Model *m) {
    int D = E3_HIDDEN;
    float *x = g_e3.scratch_x;
    float *nrm = falloc(D);
    float *attn_out = falloc(D);
    float *ffn_out = falloc(D);

    /* 1. Build fc input: [base_hidden; token_embed; prev_x] = 18432 */
    float *fc_in = g_e3.fc_in_buf;
    rmsnorm(fc_in, base_hidden, g_e3.fc_norm[0], D, c->eps, 0);
    rmsnorm(fc_in + D, token_embed, g_e3.fc_norm[1], D, c->eps, 0);
    rmsnorm(fc_in + 2*D, x, g_e3.fc_norm[2], D, c->eps, 0);

    /* fc: (out=6144, in=18432) → produces new hidden state */
    float *new_x = falloc(D);
    matmul_f(new_x, fc_in, g_e3.fc_w, 1, E3_FC_IN, D);
    memcpy(x, new_x, (size_t)D * sizeof(float));
    free(new_x);

    /* 2. Build QKV input: [x; token_embed] = 12288 */
    float *qkv_in = g_e3.qkv_in_buf;
    rmsnorm(qkv_in, x, g_e3.attn_norm, D, c->eps, 0);
    memcpy(qkv_in + D, token_embed, (size_t)D * sizeof(float));
    e3_attention(qkv_in, pos, attn_out, c);
    rmsnorm(attn_out, attn_out, g_e3.attn_norm_2, D, c->eps, 0);
    for (int i = 0; i < D; i++) x[i] += attn_out[i];

    /* 3. FFN + residual */
    rmsnorm(nrm, x, g_e3.ffn_norm, D, c->eps, 0);
    {
        float *g = falloc(E3_FFN_INTER), *u = falloc(E3_FFN_INTER), *h = falloc(D);
        /* ffn_gate: (out=18432, in=6144) */
        matmul_f(g, nrm, g_e3.ffn_gate, 1, D, E3_FFN_INTER);
        /* ffn_up: (out=18432, in=6144) */
        matmul_f(u, nrm, g_e3.ffn_up, 1, D, E3_FFN_INTER);
        for (int i = 0; i < E3_FFN_INTER; i++)
            g[i] = swiglu(g[i], u[i], c->sw_alpha, c->sw_limit);
        /* ffn_down: (out=6144, in=18432) */
        matmul_f(h, g, g_e3.ffn_down, 1, E3_FFN_INTER, D);
        memcpy(ffn_out, h, (size_t)D * sizeof(float));
        free(g); free(u); free(h);
    }
    for (int i = 0; i < D; i++) x[i] += ffn_out[i];

    /* 4. Output norm + EAGLE3's own output projection */
    rmsnorm(nrm, x, g_e3.output_norm, D, c->eps, 0);
    matmul_f(logits, nrm, g_e3.output_w, 1, D, 200064);

    /* 5. Greedy sample */
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < c->vocab; i++)
        if (logits[i] > bv) { bv = logits[i]; best = i; }

    free(nrm); free(attn_out); free(ffn_out);
    return best;
}

#endif /* COLIBRI_M3_EAGLE3_H */
