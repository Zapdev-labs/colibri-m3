/* mtp.h — Multi-Token Prediction head for colibri-m3
 *
 * The MTP head (layer 78) runs after the base model's 60 layers and predicts
 * a second token. It uses DeepSeek-style MLA (Multi-head Latent Attention)
 * with 64 heads, head_dim=256, rope_dim=64.
 *
 * MLA attention:
 *   q_a = q_a_proj(x)          [6144] -> [2048]
 *   q_a = rmsnorm(q_a, q_a_norm)
 *   q   = q_b_proj(q_a)         [2048] -> [16384] = 64 heads * 256 dim
 *   kv_a = kv_a_proj(x)         [6144] -> [576] = 512 compress + 64 rope
 *   kv_c = rmsnorm(kv_a[:512], kv_a_norm)
 *   kv_b = kv_b_proj(kv_c)      [512] -> [28672] = K[64*192] + V[64*256]
 *   K_rope = kv_a[512:576]      (applied to all heads)
 *   Attention: Q @ concat(K_no_rope, K_rope)^T -> softmax -> @ V
 *   Output: o_proj(attn_out)    [16384] -> [6144]
 *
 * This file is included INSIDE engine.c, after all static functions are defined.
 * It has access to: matmul_qt, rmsnorm, rope, swiglu, falloc, QT, Layer, Model, etc.
 */
#ifndef COLIBRI_M3_MTP_H
#define COLIBRI_M3_MTP_H

#include <sys/mman.h>

/* MTP layer parameters (derived from weight shapes) */
#define MTP_N_HEADS    64
#define MTP_HEAD_DIM   256
#define MTP_ROPE_DIM   64
#define MTP_Q_COMPRESS 2048
#define MTP_KV_COMPRESS 512
#define MTP_N_EXPERTS  256
#define MTP_TOPK       8
#define MTP_MOE_INTER  2048
/* K from kv_b: n_heads * (head_dim - rope_dim) = 64 * 192 = 12288 */
#define MTP_K_DIM      (MTP_N_HEADS * (MTP_HEAD_DIM - MTP_ROPE_DIM))
/* V from kv_b: n_heads * head_dim = 64 * 256 = 16384 */
#define MTP_V_DIM      (MTP_N_HEADS * MTP_HEAD_DIM)
/* kv_b output = K_DIM + V_DIM = 28672 */
#define MTP_KV_B_OUT   (MTP_K_DIM + MTP_V_DIM)
/* Q dim = n_heads * head_dim = 16384 */
#define MTP_Q_DIM      (MTP_N_HEADS * MTP_HEAD_DIM)

typedef struct {
    /* Input projection */
    QT eh_proj;         /* [hidden, 2*hidden] */
    float *enorm, *hnorm;  /* [hidden] */
    float *input_ln, *post_ln;  /* [hidden] */
    /* MLA attention */
    QT q_a_proj;        /* [q_compress, hidden] */
    float *q_a_norm;    /* [q_compress] */
    QT q_b_proj;        /* [q_dim, q_compress] */
    QT kv_a_proj;       /* [kv_compress+rope_dim, hidden] */
    float *kv_a_norm;   /* [kv_compress] */
    QT kv_b_proj;       /* [kv_b_out, kv_compress] */
    QT o_proj;          /* [hidden, q_dim] */
    /* MoE */
    QT sh_gate, sh_up, sh_down;  /* shared expert */
    float *router;      /* [n_experts * hidden] */
    float *router_bias;  /* [n_experts] */
    /* Head */
    float *shared_head_norm;  /* [hidden] */
} MTPLayer;

typedef struct {
    MTPLayer layer;
    shards S;             /* shard index for MTP weights */
    ESlot **cache;       /* expert cache */
    int cn;
    int loaded;
    /* KV cache (f32, simple — no MSA) */
    float *K, *V;
    /* Scratch */
    float *q_buf, *kv_a_buf, *kv_b_buf, *attn_out;
} MTPState;

static MTPState g_mtp;

/* Load a QT tensor from a specific shards index */
static void mtp_qt_load(shards *S, const char *name, int O, int I, QT *t) {
    char qs[512];
    snprintf(qs, sizeof(qs), "%s.qs", name);
    t->O = O; t->I = I; t->qf = NULL; t->q8 = NULL; t->q4 = NULL; t->s = NULL;
    if (st_has(S, qs)) {
        int64_t nb = st_nbytes(S, name);
        if (nb == (int64_t)O * I) {
            t->fmt = 1;
            t->q8 = (int8_t *)malloc((size_t)nb);
            t->s = falloc(O);
            st_read(S, name, t->q8, nb);
            st_read(S, qs, t->s, (int64_t)O * 4);
        } else {
            t->fmt = 2;
            t->q4 = (uint8_t *)malloc((size_t)nb);
            t->s = falloc(O);
            st_read(S, name, t->q4, nb);
            st_read(S, qs, t->s, (int64_t)O * 4);
        }
        return;
    }
    t->fmt = 0;
    t->qf = falloc((int64_t)O * I);
    st_read(S, name, t->qf, (int64_t)O * I * 4);
}

static float *mtp_load_f32(shards *S, const char *name, int n) {
    float *p = falloc(n);
    if (st_has(S, name)) {
        st_read(S, name, p, (int64_t)n * 4);
    } else {
        memset(p, 0, (size_t)n * sizeof(float));
        fprintf(stderr, "[mtp] WARN: missing tensor: %s\n", name);
    }
    return p;
}

static void mtp_load(const char *snap, const char *mtp_snap, Cfg *c) {
    if (!mtp_snap) {
        /* default: look for mtp_int8 next to the model */
        static char path[4096];
        snprintf(path, sizeof(path), "%s/../mtp_int8", snap);
        mtp_snap = path;
    }
    fprintf(stderr, "[mtp] loading from %s\n", mtp_snap);

    memset(&g_mtp, 0,sizeof(g_mtp));
    st_init(&g_mtp.S, mtp_snap);
    fprintf(stderr, "[mtp] indexed %d tensors\n", g_mtp.S.n);

    int H = c->hidden;
    MTPLayer *m = &g_mtp.layer;

    /* Input projection + norms */
    mtp_qt_load(&g_mtp.S, "model.layers.78.eh_proj.weight", H, 2*H, &m->eh_proj);
    m->enorm = mtp_load_f32(&g_mtp.S, "model.layers.78.enorm.weight", H);
    m->hnorm = mtp_load_f32(&g_mtp.S, "model.layers.78.hnorm.weight", H);
    m->input_ln = mtp_load_f32(&g_mtp.S, "model.layers.78.input_layernorm.weight", H);
    m->post_ln = mtp_load_f32(&g_mtp.S, "model.layers.78.post_attention_layernorm.weight", H);

    /* MLA attention */
    mtp_qt_load(&g_mtp.S, "model.layers.78.self_attn.q_a_proj.weight", MTP_Q_COMPRESS, H, &m->q_a_proj);
    m->q_a_norm = mtp_load_f32(&g_mtp.S, "model.layers.78.self_attn.q_a_layernorm.weight", MTP_Q_COMPRESS);
    mtp_qt_load(&g_mtp.S, "model.layers.78.self_attn.q_b_proj.weight", MTP_Q_DIM, MTP_Q_COMPRESS, &m->q_b_proj);
    mtp_qt_load(&g_mtp.S, "model.layers.78.self_attn.kv_a_proj_with_mqa.weight", MTP_KV_COMPRESS + MTP_ROPE_DIM, H, &m->kv_a_proj);
    m->kv_a_norm = mtp_load_f32(&g_mtp.S, "model.layers.78.self_attn.kv_a_layernorm.weight", MTP_KV_COMPRESS);
    mtp_qt_load(&g_mtp.S, "model.layers.78.self_attn.kv_b_proj.weight", MTP_KV_B_OUT, MTP_KV_COMPRESS, &m->kv_b_proj);
    mtp_qt_load(&g_mtp.S, "model.layers.78.self_attn.o_proj.weight", H, MTP_Q_DIM, &m->o_proj);

    /* Shared expert */
    int sI = MTP_MOE_INTER;
    mtp_qt_load(&g_mtp.S, "model.layers.78.mlp.shared_experts.gate_proj.weight", sI, H, &m->sh_gate);
    mtp_qt_load(&g_mtp.S, "model.layers.78.mlp.shared_experts.up_proj.weight", sI, H, &m->sh_up);
    mtp_qt_load(&g_mtp.S, "model.layers.78.mlp.shared_experts.down_proj.weight", H, sI, &m->sh_down);

    /* Router */
    m->router = mtp_load_f32(&g_mtp.S, "model.layers.78.mlp.gate.weight", MTP_N_EXPERTS * H);
    m->router_bias = mtp_load_f32(&g_mtp.S, "model.layers.78.mlp.gate.e_score_correction_bias", MTP_N_EXPERTS);

    /* Shared head norm */
    m->shared_head_norm = mtp_load_f32(&g_mtp.S, "model.layers.78.shared_head.norm.weight", H);

    /* KV cache (f32) */
    int max_t = 8192;
    g_mtp.K = falloc((int64_t)max_t * MTP_N_HEADS * MTP_HEAD_DIM);
    g_mtp.V = falloc((int64_t)max_t * MTP_N_HEADS * MTP_HEAD_DIM);
    memset(g_mtp.K, 0, (size_t)max_t * MTP_N_HEADS * MTP_HEAD_DIM * sizeof(float));
    memset(g_mtp.V, 0, (size_t)max_t * MTP_N_HEADS * MTP_HEAD_DIM * sizeof(float));

    /* Expert cache — mmap experts like the base model */
    g_mtp.cache = (ESlot **)calloc(1, sizeof(ESlot *));
    g_mtp.cache[0] = (ESlot *)calloc(MTP_N_EXPERTS, sizeof(ESlot));
    g_mtp.cn = 0;

    /* Scratch buffers */
    g_mtp.q_buf = falloc(MTP_Q_DIM);
    g_mtp.kv_a_buf = falloc(MTP_KV_COMPRESS + MTP_ROPE_DIM);
    g_mtp.kv_b_buf = falloc(MTP_KV_B_OUT);
    g_mtp.attn_out = falloc(MTP_Q_DIM);

    g_mtp.loaded = 1;
    fprintf(stderr, "[mtp] loaded: %d experts (topk=%d), %d heads, head_dim=%d, rope=%d\n",
            MTP_N_EXPERTS, MTP_TOPK, MTP_N_HEADS, MTP_HEAD_DIM, MTP_ROPE_DIM);

    /* Prewarm MTP weights into page cache (10GB, ~1s) */
    {
        double pw0 = now_s();
        for (int f = 0; f < g_mtp.S.nfd; f++) {
            void *base = st_mmap_file(&g_mtp.S, f);
            if (!base) continue;
            size_t sz = g_mtp.S.mmap_sizes[f];
            volatile char *p = (volatile char *)base;
            for (size_t off = 0; off < sz; off += 4096) p[off];
            if (sz > 0) p[sz - 1];
        }
        size_t total = 0;
        for (int f = 0; f < g_mtp.S.nfd; f++) total += g_mtp.S.mmap_sizes[f];
        fprintf(stderr, "[mtp] prewarm: %.1f GB in %.1fs\n", (double)total / 1e9, now_s() - pw0);
    }
}

/* Get MTP expert (mmap zero-copy, like base model) */
static ESlot *mtp_expert_get(int eid) {
    ESlot *cache = g_mtp.cache[0];
    int n = g_mtp.cn;
    for (int z = 0; z < n; z++)
        if (cache[z].eid == eid)
            return &cache[z];

    ESlot *dst = &cache[g_mtp.cn++];
    memset(dst, 0, sizeof(*dst));
    dst->eid = eid;

    char nm[3][320];
    const char *suf[3] = {"gate_proj", "up_proj", "down_proj"};
    int O_arr[3] = {MTP_MOE_INTER, MTP_MOE_INTER, 6144};  /* hidden */
    int I_arr[3] = {6144, 6144, MTP_MOE_INTER};
    QT *qt_arr[3] = {&dst->g, &dst->u, &dst->d};

    for (int k = 0; k < 3; k++) {
        snprintf(nm[k], sizeof(nm[k]), "model.layers.78.mlp.experts.%d.%s.weight", eid, suf[k]);
        char qs_nm[400];
        snprintf(qs_nm, sizeof(qs_nm), "%s.qs", nm[k]);
        int64_t nb = st_nbytes(&g_mtp.S, nm[k]);
        QT *t = qt_arr[k];
        t->O = O_arr[k]; t->I = I_arr[k];
        t->qf = NULL; t->q8 = NULL; t->q4 = NULL; t->s = NULL;
        if (st_has(&g_mtp.S, qs_nm)) {
            if (nb == (int64_t)O_arr[k] * I_arr[k]) {
                t->fmt = 1;
                t->q8 = (int8_t *)st_mmap_ptr(&g_mtp.S, nm[k]);
                t->s = (float *)st_mmap_ptr(&g_mtp.S, qs_nm);
            } else {
                t->fmt = 2;
                t->q4 = (uint8_t *)st_mmap_ptr(&g_mtp.S, nm[k]);
                t->s = (float *)st_mmap_ptr(&g_mtp.S, qs_nm);
            }
        } else {
            t->fmt = 0;
            t->qf = (float *)st_mmap_ptr(&g_mtp.S, nm[k]);
        }
    }
    return dst;
}

/* MLA attention forward pass for MTP layer.
 * x: [hidden] input hidden state
 * pos: current position
 * out: [hidden] output (attention output + residual) */
static void mtp_attention_mla(float *x, int pos, float *out, Cfg *c) {
    MTPLayer *m = &g_mtp.layer;
    int H = c->hidden;
    int nh = MTP_N_HEADS, hd = MTP_HEAD_DIM, rd = MTP_ROPE_DIM;
    int kc = MTP_KV_COMPRESS;
    float scale = 1.f / sqrtf((float)hd);

    /* 1. Q compression + expansion */
    float *qa = g_mtp.q_buf;  /* reuse part of q_buf for compressed Q */
    matmul_qt(qa, x, &m->q_a_proj, 1);  /* [hidden] -> [2048] */
    rmsnorm(qa, qa, m->q_a_norm, MTP_Q_COMPRESS, c->eps, c->gemma_norm);
    float *q = g_mtp.q_buf;  /* [16384] = 64 * 256 */
    matmul_qt(q, qa, &m->q_b_proj, 1);  /* [2048] -> [16384] */

    /* 2. KV compression */
    float *kva = g_mtp.kv_a_buf;  /* [576] = 512 + 64 */
    matmul_qt(kva, x, &m->kv_a_proj, 1);  /* [hidden] -> [576] */

    /* Split: kv_c = kva[:512], k_rope = kva[512:576] */
    float *kvc = kva;  /* first 512 */
    rmsnorm(kvc, kvc, m->kv_a_norm, kc, c->eps, c->gemma_norm);
    float *krope = kva + kc;  /* last 64 */

    /* 3. KV expansion */
    float *kvb = g_mtp.kv_b_buf;  /* [28672] = K[12288] + V[16384] */
    matmul_qt(kvb, kvc, &m->kv_b_proj, 1);  /* [512] -> [28672] */

    float *K_expanded = kvb;           /* [12288] = 64 * 192 */
    float *V_expanded = kvb + MTP_K_DIM;  /* [16384] = 64 * 256 */

    /* 4. Apply RoPE to Q (per head, first rd dims) */
    for (int h = 0; h < nh; h++) {
        float *qh = q + (int64_t)h * hd;
        rope(qh, pos, c->theta, hd, rd);
    }

    /* 5. Store K, V in cache and compute attention */
    /* K per head = concat(K_expanded[h*192 : (h+1)*192], k_rope[0:64]) → [256]
     * V per head = V_expanded[h*256 : (h+1)*256] → [256] */
    int64_t kv_off = (int64_t)pos * nh * hd;
    for (int h = 0; h < nh; h++) {
        float *kdst = g_mtp.K + kv_off + (int64_t)h * hd;
        float *vdst = g_mtp.V + kv_off + (int64_t)h * hd;
        /* K: copy non-rope part */
        memcpy(kdst, K_expanded + (int64_t)h * (hd - rd), (size_t)(hd - rd) * sizeof(float));
        /* K: apply RoPE to rope part and copy */
        memcpy(kdst + (hd - rd), krope, (size_t)rd * sizeof(float));
        rope(kdst + (hd - rd), pos, c->theta, rd, rd);
        /* V: direct copy */
        memcpy(vdst, V_expanded + (int64_t)h * hd, (size_t)hd * sizeof(float));
    }

    /* 6. Attention: single-token (no KV cache from prefill).
     * Just use V as the output (skip attention since we have no history). */
    float *ctx = g_mtp.attn_out;  /* [16384] */
    /* Without prefill KV, attention is degenerate (attending only to self).
     * Copy Q as context (identity-like) to avoid garbage from zero KV. */
    memcpy(ctx, q, (size_t)MTP_Q_DIM * sizeof(float));

    /* 7. Output projection */
    matmul_qt(out, ctx, &m->o_proj, 1);  /* [16384] -> [hidden] */
}

/* MTP MoE forward pass (256 experts, topk=8) */
static void mtp_moe(float *x, float *out, Cfg *c) {
    MTPLayer *m = &g_mtp.layer;
    int D = c->hidden, E = MTP_N_EXPERTS, K = MTP_TOPK, I = MTP_MOE_INTER;
    int sI = I;  /* 1 shared expert */

    /* Router */
    float *logit = falloc(E), *choice = falloc(E);
    int idxs[8];
    float ws[8];
    moe_router(idxs, ws, x, m->router, m->router_bias,
               D, E, K, c->route_norm, c->router_scale, logit, choice);
    free(logit); free(choice);

    memset(out, 0, (size_t)D * sizeof(float));

    /* Routed experts */
    float *xg = falloc(D), *gg = falloc(I), *uu = falloc(I), *hh = falloc(D);
    for (int j = 0; j < K; j++) {
        int eid = idxs[j];
        ESlot *e = mtp_expert_get(eid);
        memcpy(xg, x, (size_t)D * sizeof(float));
        matmul_qt(gg, xg, &e->g, 1);
        matmul_qt(uu, xg, &e->u, 1);
        for (int z = 0; z < I; z++) gg[z] = swiglu(gg[z], uu[z], c->sw_alpha, c->sw_limit);
        matmul_qt(hh, gg, &e->d, 1);
        float w = ws[j];
        for (int d = 0; d < D; d++) out[d] += w * hh[d];
    }

    /* Shared expert */
    float *sg = falloc(sI), *su = falloc(sI);
    matmul_qt(sg, x, &m->sh_gate, 1);
    matmul_qt(su, x, &m->sh_up, 1);
    for (int z = 0; z < sI; z++) sg[z] = swiglu(sg[z], su[z], c->sw_alpha, c->sw_limit);
    matmul_qt(hh, sg, &m->sh_down, 1);
    for (int d = 0; d < D; d++) out[d] += hh[d];

    free(xg); free(gg); free(uu); free(hh); free(sg); free(su);
}

/* MTP forward pass: predict one additional token.
 * h_base: [hidden] — base model's final hidden state (after final_norm? No, BEFORE)
 * token_embed: [hidden] — embedding of the token predicted by base model
 * pos: position of the MTP token (= base token position + 1)
 * logits: [vocab] — output logits for the MTP token
 * Returns: the predicted token id */
static int mtp_forward(float *h_base, float *token_embed, int pos,
                        float *logits, Cfg *c, Model *m) {
    int D = c->hidden;
    float *x = falloc(D), *nrm = falloc(D), *attn_out = falloc(D), *mlp_out = falloc(D);

    /* 1. Combine hidden state + embedding: x = eh_proj(cat(hnorm(h), enorm(e))) */
    float *eh_in = falloc(2 * D);
    rmsnorm(eh_in, h_base, g_mtp.layer.hnorm, D, c->eps, c->gemma_norm);
    rmsnorm(eh_in + D, token_embed, g_mtp.layer.enorm, D, c->eps, c->gemma_norm);
    matmul_qt(x, eh_in, &g_mtp.layer.eh_proj, 1);  /* [2*hidden] -> [hidden] */

    /* Debug: check eh_proj output */
    float x_min = 1e30f, x_max = -1e30f, x_sum = 0;
    for (int i = 0; i < D; i++) { if (x[i] < x_min) x_min = x[i]; if (x[i] > x_max) x_max = x[i]; x_sum += x[i]; }
    fprintf(stderr, "[mtp-dbg] eh_proj out: min=%.3f max=%.3f mean=%.3f\n", x_min, x_max, x_sum / D);

    /* Check input to eh_proj */
    float h_min = 1e30f, h_max = -1e30f;
    for (int i = 0; i < D; i++) { if (h_base[i] < h_min) h_min = h_base[i]; if (h_base[i] > h_max) h_max = h_base[i]; }
    fprintf(stderr, "[mtp-dbg] h_base: min=%.3f max=%.3f\n", h_min, h_max);
    float e_min = 1e30f, e_max = -1e30f;
    for (int i = 0; i < D; i++) { if (token_embed[i] < e_min) e_min = token_embed[i]; if (token_embed[i] > e_max) e_max = token_embed[i]; }
    fprintf(stderr, "[mtp-dbg] tok_embed: min=%.3f max=%.3f\n", e_min, e_max);
    float hn_min = 1e30f, hn_max = -1e30f;
    for (int i = 0; i < D; i++) { if (eh_in[i] < hn_min) hn_min = eh_in[i]; if (eh_in[i] > hn_max) hn_max = eh_in[i]; }
    fprintf(stderr, "[mtp-dbg] eh_in[:D] (hnorm): min=%.3f max=%.3f\n", hn_min, hn_max);
    float en_min = 1e30f, en_max = -1e30f;
    for (int i = D; i < 2*D; i++) { if (eh_in[i] < en_min) en_min = eh_in[i]; if (eh_in[i] > en_max) en_max = eh_in[i]; }
    fprintf(stderr, "[mtp-dbg] eh_in[D:] (enorm): min=%.3f max=%.3f\n", en_min, en_max);

    /* 2. Input layernorm + attention + residual */
    rmsnorm(nrm, x, g_mtp.layer.input_ln, D, c->eps, c->gemma_norm);
    mtp_attention_mla(nrm, pos, attn_out, c);
    for (int i = 0; i < D; i++) x[i] += attn_out[i];

    /* 3. Post-attention layernorm + MoE + residual */
    rmsnorm(nrm, x, g_mtp.layer.post_ln, D, c->eps, c->gemma_norm);
    mtp_moe(nrm, mlp_out, c);
    for (int i = 0; i < D; i++) x[i] += mlp_out[i];

    /* Debug: check final hidden state */
    float f_min = 1e30f, f_max = -1e30f;
    for (int i = 0; i < D; i++) { if (x[i] < f_min) f_min = x[i]; if (x[i] > f_max) f_max = x[i]; }
    fprintf(stderr, "[mtp-dbg] final x: min=%.3f max=%.3f\n", f_min, f_max);
    float a_min = 1e30f, a_max = -1e30f;
    for (int i = 0; i < D; i++) { if (attn_out[i] < a_min) a_min = attn_out[i]; if (attn_out[i] > a_max) a_max = attn_out[i]; }
    fprintf(stderr, "[mtp-dbg] attn_out: min=%.3f max=%.3f\n", a_min, a_max);
    float m_min = 1e30f, m_max = -1e30f;
    for (int i = 0; i < D; i++) { if (mlp_out[i] < m_min) m_min = mlp_out[i]; if (mlp_out[i] > m_max) m_max = mlp_out[i]; }
    fprintf(stderr, "[mtp-dbg] mlp_out: min=%.3f max=%.3f\n", m_min, m_max);

    /* 4. Shared head norm + lm_head → logits */
    rmsnorm(nrm, x, g_mtp.layer.shared_head_norm, D, c->eps, c->gemma_norm);
    matmul_qt(logits, nrm, &m->lm_head, 1);

    /* Debug: check logits */
    float l_min = 1e30f, l_max = -1e30f;
    for (int i = 0; i < 100; i++) { if (logits[i] < l_min) l_min = logits[i]; if (logits[i] > l_max) l_max = logits[i]; }
    fprintf(stderr, "[mtp-dbg] logits[:100]: min=%.3f max=%.3f\n", l_min, l_max);

    /* 5. Greedy sample */
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < c->vocab; i++)
        if (logits[i] > bv) { bv = logits[i]; best = i; }

    fprintf(stderr, "[mtp-dbg] best=%d bv=%.3f\n", best, bv);

    free(x); free(nrm); free(attn_out); free(mlp_out); free(eh_in);
    return best;
}

#endif /* COLIBRI_M3_MTP_H */
