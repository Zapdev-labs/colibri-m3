/* colibri-m3 — original MiniMax-M3 streaming MoE engine
 * Dense stack resident (int4/8). Routed experts streamed from disk with LRU.
 * GQA + partial RoPE + SwiGLU-OAI + sigmoid router. Optional PlanarQuant KV.
 * Zero runtime deps beyond libc + OpenMP. Target: big CPU boxes (AVX-512).
 */
#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/resource.h>
#endif
#ifdef __AVX512F__
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif
#include "json.h"
#include "numa.h"
#include "observability.h"
#include "planar_kv.h"
#include "st.h"
#include "vnni.h"

/* ---------- config ---------- */
typedef struct {
    int hidden, layers, heads, kv_heads, head_dim, vocab;
    int experts, topk, moe_inter, dense_inter, first_dense, n_shared;
    int rotary_dim, gemma_norm, route_norm;
    float eps, theta, router_scale, sw_alpha, sw_limit;
    int eos;
    /* MSA (MiniMax Sparse Attention) config — parsed from sparse_attention_config */
    int idx_dim;       /* sparse_index_dim, default 128 */
    int idx_heads;     /* sparse_num_index_heads, default = kv_heads */
    int topk_blk;      /* sparse_topk_blocks, default 16 */
    int blk_size;      /* sparse_block_size, default 128 */
    int init_blk;      /* sparse_init_block, default 0 (always-selected sink block) */
    int local_blk;     /* sparse_local_block, default 1 (local window blocks forced) */
    int *sparse_freq;  /* [layers] — 1 = MSA, 0 = dense GQA */
} Cfg;

/* fmt: 0 f32, 1 int8+scale, 2 int4 packed+scale */
typedef struct {
    int fmt, O, I;
    float *qf, *s;
    int8_t *q8;
    uint8_t *q4;
} QT;

typedef struct {
    float *in_ln, *post_ln, *qn, *kn, *router, *router_bias;
    QT q, k, v, o;
    QT gate, up, down;       /* dense MLP */
    QT sh_gate, sh_up, sh_down;
    int sparse;
    /* MSA indexer tensors (loaded only for layers where sparse_attention_freq[i]==1) */
    QT iq, ik;              /* index_q_proj [idx_heads*idx_dim, hidden], index_k_proj [idx_dim, hidden] */
    float *iqn, *ikn;       /* index_q_norm [idx_dim], index_k_norm [idx_dim] (f32) */
    int msa;                /* 1 = use MSA sparse attention, 0 = dense GQA */
} Layer;

typedef struct {
    int eid;
    QT g, u, d;
    uint8_t *slab;
    uint64_t used;
} ESlot;

typedef struct {
    Cfg c;
    shards S;
    int ebits, dbits, ecap;
    QT embed, lm_head;
    float *final_norm;
    Layer *L;
    ESlot **cache;
    int *cn;
    float **K, **V;
    int8_t **Kq, **Vq;
    float **Ks, **Vs;
    int max_t, kv_i8, planar;
    uint64_t clock, hits, miss;
    /* MSA indexer K cache: per-layer [max_t * idx_dim] f32 */
    float **Kidx;
    int msa_layers;  /* count of layers with MSA enabled */
} Model;

static float g_temp = 1.0f, g_topp = 0.95f;
static int g_topk_samp = 40, g_planar_bits = 3;

/* f8-oracle-validation: teacher-forcing, NaN/debug-trace, and seeded-determinism
 * knobs. All are opt-in via env vars / CLI flags; defaults preserve the existing
 * behavior (greedy when TEMP<=0, no TF dump, no nan instrumentation, time-seeded). */
static int g_seed = 0;            /* 0 = time-seeded (legacy); >0 = fixed seed */
static int g_tf_mode = 0;         /* 1 = teacher-force mode: dump top-K logprobs per position */
static int g_moe_parallel = 0;   /* f12: 1 = parallel MoE expert dispatch */
static int g_use_avx512_i4 = 0;
static int g_use_mtp = 0;
static const char *g_e3_dir = NULL; /* opt-in AVX-512 i4 FMA kernel (slower for memory-bound workloads) */
static int g_bench_mode = 0;     /* f13: 1 = native C benchmark mode (per-token timing + JSON summary) */
static NumaTopo g_numa;          /* f10: discovered NUMA topology (zeroed until numa_discover) */
static int g_tf_topk = 200;       /* top-K logprobs to dump per TF position (matches oracle) */
static const char *g_tf_out_path = NULL;  /* TF dump file (default: stdout) */
static int g_nan_check = 0;       /* 1 = M3_CHECK_NAN mode: assert no NaN/Inf in intermediates */
static int g_debug_trace = 0;    /* 1 = DEBUG_TRACE mode: dump per-tensor stats (min/max/mean) */
static long g_nan_failures = 0;   /* count of NaN/Inf failures detected */

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static float *falloc(int64_t n) {
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) {
        fprintf(stderr, "OOM\n");
        exit(1);
    }
    return p;
}

#ifdef __AVX512F__
static inline float hsum512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
static inline float dot_f(const float *a, const float *b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float s = hsum512(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline void axpy_f(float *y, const float *x, float w, int n) {
    __m512 wv = _mm512_set1_ps(w);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 yv = _mm512_loadu_ps(y + i);
        yv = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), wv, yv);
        _mm512_storeu_ps(y + i, yv);
    }
    for (; i < n; i++) y[i] += w * x[i];
}
#elif defined(__AVX2__)
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}
static inline float dot_f(const float *a, const float *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    float s = hsum256(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline void axpy_f(float *y, const float *x, float w, int n) {
    __m256 wv = _mm256_set1_ps(w);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 yv = _mm256_loadu_ps(y + i);
        yv = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), wv, yv);
        _mm256_storeu_ps(y + i, yv);
    }
    for (; i < n; i++) y[i] += w * x[i];
}
#else
static inline float dot_f(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline void axpy_f(float *y, const float *x, float w, int n) {
    for (int i = 0; i < n; i++) y[i] += w * x[i];
}
#endif

static inline float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

/* SwiGLUOAI activation matching HF MiniMaxM3VLDenseMLP._apply_gate /
 * MiniMaxM3VLExperts._apply_gate exactly:
 *   gate = gate.clamp(max=lim)            (asymmetric: upper-only clamp)
 *   up   = up.clamp(min=-lim, max=lim)    (symmetric clamp on up)
 *   glu  = gate * sigmoid(gate * alpha)
 *   out  = (up + 1.0) * glu
 * The +1.0 multiplier on `up` and the asymmetric (max-only) gate clamp are
 * what distinguish SwiGLUOAI from a plain clamp(gate)*sigmoid(alpha*gate)*up.
 * Reference: transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py. */
static inline float swiglu(float gate, float up, float a, float lim) {
    float g = gate > lim ? lim : gate;                      /* clamp(max=lim) */
    float u = up > lim ? lim : (up < -lim ? -lim : up);     /* clamp([-lim,lim]) */
    float glu = g * sigmoidf_(a * g);
    return (u + 1.0f) * glu;
}

static void rmsnorm(float *o, const float *x, const float *w, int D, float eps, int gemma) {
    double ms = 0;
    for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    if (gemma)
        for (int i = 0; i < D; i++) o[i] = x[i] * r * (1.f + w[i]);
    else
        for (int i = 0; i < D; i++) o[i] = x[i] * r * w[i];
}
static void softmax(float *x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; i++)
        if (x[i] > m) m = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* f8: NaN/Inf check + optional tensor-stats dump for M3_CHECK_NAN / DEBUG_TRACE.
 * Called from layer_fwd / run_gen / run_teacher_force at every intermediate tensor.
 * - M3_CHECK_NAN=1: prints "nan-check: L<layer> <kernel> OK" per (layer, kernel)
 *   pair, or "nan-check: FAILED ..." on the first NaN/Inf element. The contract
 *   (VAL-CORR-024) requires zero FAILED lines over a 50-token run.
 * - DEBUG_TRACE=1: additionally prints per-tensor min/max/mean stats (the
 *   --debug-trace mode the feature spec requests for dumping tensor stats).
 * Both modes are opt-in via env vars; the default path incurs no overhead. */
static void nan_check_layer(int layer, const char *kernel, const float *p, int64_t n) {
    if (!g_nan_check && !g_debug_trace) return;
    for (int64_t i = 0; i < n; i++) {
        if (isnan(p[i]) || isinf(p[i])) {
            fprintf(stderr, "nan-check: FAILED L%d %s at idx %lld val=%g\n",
                    layer, kernel, (long long)i, (double)p[i]);
            g_nan_failures++;
            return;
        }
    }
    if (g_debug_trace) {
        float mn = p[0], mx = p[0];
        double sum = 0.0;
        for (int64_t i = 0; i < n; i++) {
            float v = p[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += (double)v;
        }
        fprintf(stderr, "trace: L%d %-28s n=%lld min=%.6g max=%.6g mean=%.6g\n",
                layer, kernel, (long long)n, (double)mn, (double)mx,
                sum / (double)n);
    } else {
        fprintf(stderr, "nan-check: L%d %-28s OK\n", layer, kernel);
    }
}

/* f8: logsoftmax + top-K selection over the full vocabulary.
 * Computes logsoftmax(logits) over V entries, then selects the top-K token IDs
 * by logprob (descending). Used by teacher-forcing mode to dump per-position
 * top-K logprobs in the same format as the oracle's `per_position[].top` array
 * (each entry is [token_id, logprob]). O(K*V) — ~40M comparisons for K=200,
 * V=200064, ~50ms per position on the remote Xeon. */
static void logsoftmax_topk(const float *logits, int V, int K,
                            int *out_ids, float *out_lps) {
    if (V <= 0 || K <= 0) return;
    float mx = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < V; i++) sum += exp((double)(logits[i] - mx));
    float lse = (float)((double)mx + log(sum));
    size_t Vs = (size_t)V;
    char *taken = (char *)calloc(Vs > 0 ? Vs : 1, 1);
    int kmax = K < V ? K : V;
    for (int k = 0; k < kmax; k++) {
        int best = -1;
        float bv = -1e30f;
        for (int i = 0; i < V; i++) {
            if (!taken[i] && logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        }
        if (best < 0) break;
        taken[best] = 1;
        out_ids[k] = best;
        out_lps[k] = logits[best] - lse;
    }
    free(taken);
}

/* Partial rotate-half RoPE on first rotary_dim channels */
static void rope(float *x, int pos, float theta, int hd, int rd) {
    if (rd <= 0 || rd > hd) rd = hd;
    int h2 = rd / 2;
    float tmp[256];
    if (rd > 256) {
        fprintf(stderr, "rotary_dim too large\n");
        exit(1);
    }
    memcpy(tmp, x, (size_t)rd * sizeof(float));
    for (int j = 0; j < h2; j++) {
        float inv = powf(theta, -2.f * j / (float)rd);
        float ang = pos * inv, c = cosf(ang), s = sinf(ang);
        float x1 = tmp[j], x2 = tmp[j + h2];
        x[j] = x1 * c - x2 * s;
        x[j + h2] = x2 * c + x1 * s;
    }
}

static void matmul_f(float *y, const float *x, const float *W, int S, int I, int O) {
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            y[(int64_t)s * O + o] = dot_f(x + (int64_t)s * I, w, I);
        }
    }
}

/* INT8 KV cache quantize/dequantize helpers — symmetric per-row scaling.
 * Extracted so unit tests (VAL-CORR-026) can exercise the round-trip without
 * driving a full attention pass. scale = max(|x|) / 127; q = round(x / scale). */
static inline float kv_quantize_i8(const float *x, int8_t *q, int n) {
    float mx = 1e-8f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > mx) mx = a;
    }
    float scale = mx / 127.f;
    float inv = 1.f / scale;
    for (int i = 0; i < n; i++) q[i] = (int8_t)lrintf(x[i] * inv);
    return scale;
}
static inline void kv_dequantize_i8(float *out, const int8_t *q, float scale, int n) {
    for (int i = 0; i < n; i++) out[i] = (float)q[i] * scale;
}

static void matmul_i8(float *y, const float *x, const int8_t *q, const float *sc, int S, int I, int O) {
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float scale = sc[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0;
            for (int i = 0; i < I; i++) a += xs[i] * (float)w[i];
            y[(int64_t)s * O + o] = a * scale;
        }
    }
}
static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *sc, int S, int I, int O) {
    int rb = (I + 1) / 2;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        float scale = sc[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0;
            int i = 0;
            for (; i + 1 < I; i += 2) {
                uint8_t b = w[i >> 1];
                a += xs[i] * (float)((int)(b & 0xF) - 8) + xs[i + 1] * (float)((int)(b >> 4) - 8);
            }
            if (i < I) a += xs[i] * (float)((int)(w[i >> 1] & 0xF) - 8);
            y[(int64_t)s * O + o] = a * scale;
        }
    }
}

/* Group-scaled int4 matmul: sc is [O * ngroups] where ngroups = ceil(I/gs).
 * Each (output row o, input group g) has its own scale — used by VAL-CORR-008
 * to exercise the group-scaled (fmt=4, gs=128) kernel variant. The per-row
 * matmul_i4 above is the special case gs=I. */
static void matmul_i4_grouped(float *y, const float *x, const uint8_t *q4,
                              const float *sc, int S, int I, int O, int gs) {
    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        const float *sco = sc + (int64_t)o * ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0;
            for (int g = 0; g < ng; g++) {
                int gstart = g * gs;
                int gend = gstart + gs;
                if (gend > I) gend = I;
                float ga = 0;
                int i = gstart;
                for (; i + 1 < gend; i += 2) {
                    uint8_t b = w[i >> 1];
                    ga += xs[i] * (float)((int)(b & 0xF) - 8) +
                          xs[i + 1] * (float)((int)(b >> 4) - 8);
                }
                if (i < gend) ga += xs[i] * (float)((int)(w[i >> 1] & 0xF) - 8);
                a += ga * sco[g];
            }
            y[(int64_t)s * O + o] = a;
        }
    }
}

static void matmul_qt(float *y, const float *x, QT *w, int S) {
    if (w->fmt == 0) matmul_f(y, x, w->qf, S, w->I, w->O);
    else if (w->fmt == 1) {
        if (g_use_vnni) matmul_i8_vnni(y, x, w->q8, w->s, S, w->I, w->O);
        else            matmul_i8(y, x, w->q8, w->s, S, w->I, w->O);
    } else {
        if (g_use_vnni)      matmul_i4_vnni(y, x, w->q4, w->s, S, w->I, w->O);
        else if (g_use_avx512_i4) matmul_i4_avx512(y, x, w->q4, w->s, S, w->I, w->O);
        else                matmul_i4(y, x, w->q4, w->s, S, w->I, w->O);
    }
}

static void qt_load(Model *m, const char *name, int O, int I, int bits, QT *t) {
    char qs[512];
    snprintf(qs, sizeof(qs), "%s.qs", name);
    t->O = O;
    t->I = I;
    t->qf = NULL;
    t->q8 = NULL;
    t->q4 = NULL;
    t->s = NULL;
    if (st_has(&m->S, qs)) {
        int64_t nb = st_nbytes(&m->S, name);
        if (nb == (int64_t)O * I) {
            t->fmt = 1;
            t->q8 = (int8_t *)malloc((size_t)nb);
            t->s = falloc(O);
            st_read(&m->S, name, t->q8, nb);
            st_read(&m->S, qs, t->s, (int64_t)O * 4);
        } else {
            t->fmt = 2;
            t->q4 = (uint8_t *)malloc((size_t)nb);
            t->s = falloc(O);
            st_read(&m->S, name, t->q4, nb);
            st_read(&m->S, qs, t->s, (int64_t)O * 4);
        }
        (void)bits;
        return;
    }
    /* f32 fallback */
    t->fmt = 0;
    t->qf = falloc((int64_t)O * I);
    st_read(&m->S, name, t->qf, (int64_t)O * I * 4);
}

static int gi(jval *r, const char *k) {
    jval *v = json_get(r, k);
    return v ? (int)v->num : 0;
}
static float gf(jval *r, const char *k, float d) {
    jval *v = json_get(r, k);
    return v ? (float)v->num : d;
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) {
    }
    b[n] = 0;
    fclose(f);
    char *ar = NULL;
    jval *root = json_parse(b, &ar);
    free(b);
    jval *tc = json_get(root, "text_config");
    jval *r = tc ? tc : root;
    c->hidden = gi(r, "hidden_size");
    c->layers = gi(r, "num_hidden_layers");
    c->heads = gi(r, "num_attention_heads");
    c->kv_heads = gi(r, "num_key_value_heads");
    c->experts = gi(r, "num_experts");
    if (!c->experts) c->experts = gi(r, "num_local_experts");
    c->topk = gi(r, "num_experts_per_tok");
    c->moe_inter = gi(r, "moe_intermediate_size");
    if (!c->moe_inter) c->moe_inter = gi(r, "intermediate_size");
    c->dense_inter = gi(r, "dense_intermediate_size");
    if (!c->dense_inter) c->dense_inter = gi(r, "intermediate_size");
    c->first_dense = gi(r, "first_k_dense_replace");
    c->head_dim = gi(r, "head_dim");
    if (!c->head_dim) c->head_dim = c->hidden / c->heads;
    c->n_shared = gi(r, "num_shared_experts");
    if (!c->n_shared) c->n_shared = 1;
    c->vocab = gi(r, "vocab_size");
    c->rotary_dim = gi(r, "rotary_dim");
    if (!c->rotary_dim) c->rotary_dim = c->head_dim / 2;
    c->eps = gf(r, "rms_norm_eps", 1e-6f);
    c->theta = gf(r, "rope_theta", 5000000.f);
    jval *rp = json_get(r, "rope_parameters");
    if (rp) {
        jval *th = json_get(rp, "rope_theta");
        if (th) c->theta = (float)th->num;
    }
    c->router_scale = gf(r, "router_scaling_factor", 0.f);
    if (c->router_scale == 0.f) c->router_scale = gf(r, "routed_scaling_factor", 2.f);
    c->sw_alpha = gf(r, "swiglu_alpha", 1.702f);
    c->sw_limit = gf(r, "swiglu_limit", 7.f);
    jval *gn = json_get(r, "use_gemma_norm");
    c->gemma_norm = (gn && gn->t == J_BOOL) ? gn->boolean : 1;
    jval *rn = json_get(r, "route_norm");
    c->route_norm = (rn && rn->t == J_BOOL) ? rn->boolean : 1;
    jval *eo = json_get(r, "eos_token_id");
    if (!eo) eo = json_get(root, "eos_token_id");
    c->eos = eo ? (int)eo->num : 200020;

    /* MSA: parse sparse_attention_config (from HF config) */
    c->idx_dim = 0;
    c->idx_heads = 0;
    c->topk_blk = 16;
    c->blk_size = 128;
    c->init_blk = 0;
    c->local_blk = 1;
    c->sparse_freq = NULL;
    jval *sac = json_get(r, "sparse_attention_config");
    if (sac) {
        c->idx_dim = gi(sac, "sparse_index_dim");
        c->idx_heads = gi(sac, "sparse_num_index_heads");
        c->topk_blk = gi(sac, "sparse_topk_blocks");
        c->blk_size = gi(sac, "sparse_block_size");
        c->init_blk = gi(sac, "sparse_init_block");
        c->local_blk = gi(sac, "sparse_local_block");
        jval *freq = json_get(sac, "sparse_attention_freq");
        if (freq && freq->t == J_ARR && freq->len > 0) {
            c->sparse_freq = (int *)calloc((size_t)c->layers, sizeof(int));
            int ncopy = freq->len < c->layers ? freq->len : c->layers;
            for (int i = 0; i < ncopy; i++) c->sparse_freq[i] = (int)freq->kids[i]->num;
        }
    }
    /* defaults if sparse_attention_config absent */
    if (!c->idx_dim) c->idx_dim = 128;
    if (!c->idx_heads) c->idx_heads = c->kv_heads;
    if (!c->topk_blk) c->topk_blk = 16;
    if (!c->blk_size) c->blk_size = 128;
    if (c->init_blk < 0) c->init_blk = 0;
    if (c->local_blk < 0) c->local_blk = 1;
    int msa_count = 0;
    if (c->sparse_freq) {
        for (int i = 0; i < c->layers; i++) msa_count += c->sparse_freq[i] ? 1 : 0;
    }
    fprintf(stderr,
            "[cfg] M3 H=%d L=%d heads=%d/%d experts=%d topk=%d rot=%d first_dense=%d\n",
            c->hidden, c->layers, c->heads, c->kv_heads, c->experts, c->topk, c->rotary_dim,
            c->first_dense);
    fprintf(stderr,
            "[cfg] MSA idx_dim=%d idx_heads=%d topk_blk=%d blk_size=%d init=%d local=%d msa_layers=%d/%d\n",
            c->idx_dim, c->idx_heads, c->topk_blk, c->blk_size, c->init_blk, c->local_blk,
            msa_count, c->layers);
}

static float *load_f32(Model *m, const char *name, int n) {
    float *p = falloc(n);
    st_read(&m->S, name, p, (int64_t)n * 4);
    return p;
}

static void load_model(Model *m, const char *snap, int ebits, int dbits, int ecap, int max_t) {
    memset(m, 0, sizeof(*m));
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    m->ebits = ebits;
    m->dbits = dbits;
    m->ecap = ecap;
    m->max_t = max_t;
    m->kv_i8 = getenv("KV_I8") ? atoi(getenv("KV_I8")) : 0;
    m->planar = getenv("PLANAR_KV") ? atoi(getenv("PLANAR_KV")) : 0;
    if (m->planar) m->kv_i8 = 1;
    /* f14: auto-enable INT8 KV when context >= 16K so the cache fits in the
     * 350 GB RAM budget alongside the 199 GB int4 weights. At 32K context
     * INT8 KV is ~2 GB; at 64K it is ~4 GB; the BF16 path at 32K is ~4 GB
     * and at 1M would be ~125 GB (out of scope). User-set KV_I8 wins. */
    if (!getenv("KV_I8") && !m->planar && max_t >= 16384) {
        m->kv_i8 = 1;
        fprintf(stderr, "[f14] auto-enabling INT8 KV cache (ctx=%d >= 16K)\n", max_t);
    }
    Cfg *c = &m->c;
    int D = c->hidden;
    qt_load(m, "model.embed_tokens.weight", c->vocab, D, dbits, &m->embed);
    if (st_has(&m->S, "lm_head.weight"))
        qt_load(m, "lm_head.weight", c->vocab, D, dbits, &m->lm_head);
    else
        m->lm_head = m->embed;
    m->final_norm = load_f32(m, "model.norm.weight", D);
    m->L = (Layer *)calloc((size_t)c->layers, sizeof(Layer));
    m->cache = (ESlot **)calloc((size_t)c->layers, sizeof(ESlot *));
    m->cn = (int *)calloc((size_t)c->layers, sizeof(int));
    char nm[384], nm2[384];
    double t0 = now_s();
    for (int i = 0; i < c->layers; i++) {
        Layer *l = &m->L[i];
        l->sparse = (i >= c->first_dense);
        snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", i);
        l->in_ln = load_f32(m, nm, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", i);
        l->post_ln = load_f32(m, nm, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.weight", i);
        qt_load(m, nm, c->heads * c->head_dim, D, dbits, &l->q);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.weight", i);
        qt_load(m, nm, c->kv_heads * c->head_dim, D, dbits, &l->k);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.weight", i);
        qt_load(m, nm, c->kv_heads * c->head_dim, D, dbits, &l->v);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", i);
        qt_load(m, nm, D, c->heads * c->head_dim, dbits, &l->o);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_norm.weight", i);
        if (st_has(&m->S, nm)) l->qn = load_f32(m, nm, c->head_dim);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_norm.weight", i);
        if (st_has(&m->S, nm)) l->kn = load_f32(m, nm, c->head_dim);
        /* MSA: load index_q_proj, index_k_proj, index_q_norm, index_k_norm for
         * layers where sparse_attention_freq[i] == 1 (layers 3-59 in MiniMax M3). */
        l->msa = (c->sparse_freq && c->sparse_freq[i]) ? 1 : 0;
        if (l->msa) {
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.index_q_proj.weight", i);
            qt_load(m, nm, c->idx_heads * c->idx_dim, D, dbits, &l->iq);
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.index_k_proj.weight", i);
            qt_load(m, nm, c->idx_dim, D, dbits, &l->ik);
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.index_q_norm.weight", i);
            if (st_has(&m->S, nm)) l->iqn = load_f32(m, nm, c->idx_dim);
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.index_k_norm.weight", i);
            if (st_has(&m->S, nm)) l->ikn = load_f32(m, nm, c->idx_dim);
            m->msa_layers++;
        }
        if (l->sparse) {
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.weight", i);
            l->router = load_f32(m, nm, c->experts * D);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.e_score_correction_bias", i);
            if (!st_has(&m->S, nm))
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.expert_bias", i);
            if (st_has(&m->S, nm))
                l->router_bias = load_f32(m, nm, c->experts);
            else {
                l->router_bias = falloc(c->experts);
                memset(l->router_bias, 0, (size_t)c->experts * sizeof(float));
            }
            int sI = c->moe_inter * c->n_shared;
            snprintf(nm2, sizeof(nm2), "model.layers.%d.mlp.shared_experts.gate_proj.weight", i);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_mlp.gate_proj.weight", i);
            qt_load(m, st_has(&m->S, nm2) ? nm2 : nm, sI, D, dbits, &l->sh_gate);
            snprintf(nm2, sizeof(nm2), "model.layers.%d.mlp.shared_experts.up_proj.weight", i);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_mlp.up_proj.weight", i);
            qt_load(m, st_has(&m->S, nm2) ? nm2 : nm, sI, D, dbits, &l->sh_up);
            snprintf(nm2, sizeof(nm2), "model.layers.%d.mlp.shared_experts.down_proj.weight", i);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_mlp.down_proj.weight", i);
            qt_load(m, st_has(&m->S, nm2) ? nm2 : nm, D, sI, dbits, &l->sh_down);
            m->cache[i] = (ESlot *)calloc((size_t)ecap, sizeof(ESlot));
        } else {
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", i);
            qt_load(m, nm, c->dense_inter, D, dbits, &l->gate);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", i);
            qt_load(m, nm, c->dense_inter, D, dbits, &l->up);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", i);
            qt_load(m, nm, D, c->dense_inter, dbits, &l->down);
        }
        if ((i + 1) % 10 == 0)
            fprintf(stderr, "[load] layer %d/%d  RSS-ish elapsed %.1fs\n", i + 1, c->layers, now_s() - t0);
    }
    int64_t slot = (int64_t)c->kv_heads * max_t * c->head_dim;
    if (m->kv_i8) {
        m->Kq = (int8_t **)calloc((size_t)c->layers, sizeof(int8_t *));
        m->Vq = (int8_t **)calloc((size_t)c->layers, sizeof(int8_t *));
        m->Ks = (float **)calloc((size_t)c->layers, sizeof(float *));
        m->Vs = (float **)calloc((size_t)c->layers, sizeof(float *));
        for (int i = 0; i < c->layers; i++) {
            m->Kq[i] = (int8_t *)malloc((size_t)slot);
            m->Vq[i] = (int8_t *)malloc((size_t)slot);
            m->Ks[i] = falloc((int64_t)c->kv_heads * max_t);
            m->Vs[i] = falloc((int64_t)c->kv_heads * max_t);
        }
    } else {
        m->K = (float **)calloc((size_t)c->layers, sizeof(float *));
        m->V = (float **)calloc((size_t)c->layers, sizeof(float *));
        for (int i = 0; i < c->layers; i++) {
            m->K[i] = falloc(slot);
            m->V[i] = falloc(slot);
        }
    }
    /* MSA indexer K cache: per-layer [max_t * idx_dim] f32 (one 128-dim row per KV
     * position; shared across all index heads — the K projection is single-head). */
    if (m->msa_layers > 0) {
        m->Kidx = (float **)calloc((size_t)c->layers, sizeof(float *));
        int64_t idx_slot = (int64_t)max_t * c->idx_dim;
        for (int i = 0; i < c->layers; i++) {
            if (m->L[i].msa) m->Kidx[i] = falloc(idx_slot);
        }
    }
    /* Prewarm: mmap all shard files and touch every page to pull the
     * entire model into OS page cache. With 368GB RAM and 212GB model,
     * everything fits. This eliminates page faults during inference. */
    if (!getenv("SKIP_PREWARM")) {
        double pw0 = now_s();
        for (int f = 0; f < m->S.nfd; f++) {
            void *base = st_mmap_file(&m->S, f);
            if (!base) continue;
            size_t sz = m->S.mmap_sizes[f];
            /* Touch first byte of each 4KB page to trigger page-in */
            volatile char *p = (volatile char *)base;
            for (size_t off = 0; off < sz; off += 4096)
                p[off];
            /* Touch last page too */
            if (sz > 0) p[sz - 1];
        }
        fprintf(stderr, "[prewarm] touched %.1f GB in %.1fs\n",
                (double)m->S.mmap_sizes[0] / 1e9 * 0 /* dummy */, now_s() - pw0);
        /* Calculate total mmaped */
        size_t total = 0;
        for (int f = 0; f < m->S.nfd; f++) total += m->S.mmap_sizes[f];
        fprintf(stderr, "[prewarm] %.1f GB in page cache (%.1fs)\n",
                (double)total / 1e9, now_s() - pw0);
    }
    fprintf(stderr, "[load] ready in %.1fs | expert cache %d/layer | kv_%s | msa_layers=%d\n",
            now_s() - t0, ecap,
            m->planar ? "planar" : (m->kv_i8 ? "i8" : "f32"), m->msa_layers);
    /* f10: interleave the model struct's pages across all NUMA nodes so the
     * dense-resident tensors are spread evenly between both sockets. The KV
     * cache and per-layer tensors are also covered. Best-effort: mbind failure
     * is logged but does not abort (the engine still runs correctly, just
     * with whatever locality the allocator gave us). */
    if (g_numa.interleave && g_numa.n_nodes > 1) {
        int rc = numa_interleave_memory(m, sizeof(*m), g_numa.n_nodes);
        if (rc != 0) fprintf(stderr, "[f10] WARN: mbind interleave rc=%d\n", rc);
    }
    /* f15: emit load_complete telemetry. */
    long rss_kb = 0;
#if defined(__linux__)
    {
        FILE *f = fopen("/proc/self/status", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (sscanf(line, "VmRSS: %ld kB", &rss_kb) == 1) break;
            }
            fclose(f);
        }
    }
#endif
    telem_load_complete(c->layers, c->experts, rss_kb, m->msa_layers, m->kv_i8, m->planar);
}

static void expert_load(Model *m, int layer, int eid, ESlot *s) {
    Cfg *c = &m->c;
    char nm[3][320], qs_nm[3][320];
    const char *suf[3] = {"gate_proj", "up_proj", "down_proj"};
    int O[3] = {c->moe_inter, c->moe_inter, c->hidden};
    int I[3] = {c->hidden, c->hidden, c->moe_inter};
    QT *qt[3] = {&s->g, &s->u, &s->d};
    s->eid = eid;
    for (int k = 0; k < 3; k++) {
        snprintf(nm[k], sizeof(nm[k]), "model.layers.%d.mlp.experts.%d.%s.weight", layer, eid, suf[k]);
        snprintf(qs_nm[k], sizeof(qs_nm[k]), "%s.qs", nm[k]);
        int64_t nb = st_nbytes(&m->S, nm[k]);
        QT *t = qt[k];
        t->O = O[k]; t->I = I[k];
        t->qf = NULL; t->q8 = NULL; t->q4 = NULL; t->s = NULL;
        if (st_has(&m->S, qs_nm[k])) {
            if (nb == (int64_t)O[k] * I[k]) {
                t->fmt = 1;
                t->q8 = (int8_t *)st_mmap_ptr(&m->S, nm[k]);
                t->s = (float *)st_mmap_ptr(&m->S, qs_nm[k]);
            } else {
                t->fmt = 2;
                t->q4 = (uint8_t *)st_mmap_ptr(&m->S, nm[k]);
                t->s = (float *)st_mmap_ptr(&m->S, qs_nm[k]);
            }
        } else {
            t->fmt = 0;
            t->qf = (float *)st_mmap_ptr(&m->S, nm[k]);
        }
    }
    s->used = ++m->clock;
}

static ESlot *expert_get(Model *m, int layer, int eid) {
    ESlot *Sl = m->cache[layer];
    int n = m->cn[layer];
    for (int z = 0; z < n; z++)
        if (Sl[z].eid == eid) {
            m->hits++;
            Sl[z].used = ++m->clock;
            return &Sl[z];
        }
    m->miss++;
    ESlot *dst;
    if (n < m->ecap) {
        dst = &Sl[m->cn[layer]++];
        memset(dst, 0, sizeof(*dst));
    } else {
        int lru = 0;
        for (int z = 1; z < n; z++)
            if (Sl[z].used < Sl[lru].used) lru = z;
        dst = &Sl[lru];
        /* mmap'd experts: no free needed, data lives in mmap region */
        memset(dst, 0, sizeof(*dst));
    }
    expert_load(m, layer, eid, dst);
    return dst;
}

/* ---------- MSA (MiniMax Sparse Attention) ----------
 * Ported from llama.cpp-minimax-m3-rq: msa_block_mask_op (top-k partial_sort with
 * position-anchored local-force bias) + the per-layer ggml_flash_attn_ext dispatch.
 * Pure C, f32 KV path. The indexer projects Q/K to 128-dim, applies per-head RMSNorm,
 * then computes a max score over each 128-token KV block. Top-16 blocks are selected
 * per GQA group (init_block=0 + local_block=1 are always forced), then a block-sparse
 * softmax attention runs over the gathered tokens. sparse_attention_freq[layer]==0
 * routes through the dense GQA path (layers 0-2 in MiniMax M3).
 */

/* Compute per-block max-scores from indexer Q and the indexer K cache.
 * iq:       [idx_heads * idx_dim]  — indexer Q for the current token (RoPE/norm already applied)
 * Kidx:     [max_t * idx_dim]      — indexer K cache, one 128-dim row per KV position
 * pos:      current query position (0-based); only tokens 0..pos are visible (causal)
 * scores:   output [idx_heads * n_blocks]
 */
static void msa_indexer_scores(const float *iq, const float *Kidx, int pos,
                               int idx_heads, int idx_dim, int max_t,
                               int blk_size, int n_blocks, float *scores) {
    for (int h = 0; h < idx_heads; h++) {
        const float *iqh = iq + (int64_t)h * idx_dim;
        float *outh = scores + (int64_t)h * n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            int bstart = b * blk_size;
            int bend = bstart + blk_size;
            if (bend > pos + 1) bend = pos + 1;     /* causal: token pos inclusive */
            if (bstart >= max_t) bstart = max_t;     /* beyond cache */
            float mx = -1e30f;
            for (int t = bstart; t < bend; t++) {
                const float *kt = Kidx + (int64_t)t * idx_dim;
                float s = dot_f(iqh, kt, idx_dim);
                if (s > mx) mx = s;
            }
            outh[b] = (bend > bstart) ? mx : -1e30f;
        }
    }
}

/* Top-K block selection per GQA group with init_block + local_block forced.
 * scores:   [n_blocks] per-block max-score for one index head
 * selected: output [topk] — selected block indices (ascending order not guaranteed;
 *           forced blocks come first, then top-scoring non-forced blocks).
 * Returns the number of valid selected blocks (== topk unless n_blocks < topk).
 */
static int msa_topk_select(const float *scores, int n_blocks, int topk,
                           int init_blk, int pos, int blk_size, int local_blks,
                           int *selected) {
    char forced[8192];
    if (n_blocks > (int)sizeof(forced)) n_blocks = (int)sizeof(forced); /* safety */
    memset(forced, 0, (size_t)n_blocks);
    int local_blk = pos / blk_size;
    if (init_blk >= 0 && init_blk < n_blocks) forced[init_blk] = 1;
    for (int l = 0; l < local_blks; l++) {
        int b = local_blk - l;
        if (b >= 0 && b < n_blocks) forced[b] = 1;
    }
    int ns = 0;
    /* forced blocks first */
    for (int b = 0; b < n_blocks && ns < topk; b++) {
        if (forced[b]) selected[ns++] = b;
    }
    /* fill remaining slots with highest-scoring non-forced blocks (selection sort) */
    while (ns < topk) {
        int best = -1;
        float bv = -1e30f;
        for (int b = 0; b < n_blocks; b++) {
            if (!forced[b] && scores[b] > bv) {
                bv = scores[b];
                best = b;
            }
        }
        if (best < 0) break;      /* fewer than topk selectable blocks (small ctx) */
        forced[best] = 1;
        selected[ns++] = best;
    }
    /* pad with -1 if n_blocks < topk */
    while (ns < topk) selected[ns++] = -1;
    return ns;
}

/* Read one K (or V) row from the KV cache into a dense f32 buffer.
 * Handles f32, i8, and planar-i8 cache formats. */
static inline void msa_kv_read(Model *m, int layer, int kv_head, int t,
                               float *out, int hd, int is_v) {
    int64_t off = ((int64_t)kv_head * m->max_t + t) * hd;
    int64_t soff = (int64_t)kv_head * m->max_t + t;
    if (m->kv_i8) {
        int8_t *q = is_v ? m->Vq[layer] : m->Kq[layer];
        float *sc = is_v ? m->Vs[layer] : m->Ks[layer];
        if (m->planar) {
            planar_kv_decode(out, q + off, sc[soff], hd, g_planar_bits);
        } else {
            float s = sc[soff];
            for (int i = 0; i < hd; i++) out[i] = (float)q[off + i] * s;
        }
    } else {
        float *p = is_v ? m->V[layer] : m->K[layer];
        memcpy(out, p + off, (size_t)hd * sizeof(float));
    }
}

/* Block-sparse softmax attention for one GQA group, single query token.
 * Q:        [n_q_in_group * head_dim]  — Q vectors for the n_q_in_group heads in this group
 * selected: [topk]                     — selected block indices (from msa_topk_select)
 * out:      [n_q_in_group * head_dim]  — attention output (zeroed on entry)
 */
static float *g_msa_kbuf=NULL,*g_msa_vbuf=NULL,*g_msa_sc=NULL;
static int g_msa_buf_sz=0;
static void msa_group_attention(Model *m, int layer, int kv_head,
                                const float *Q, const int *selected, int topk,
                                int n_q_in_group, int head_dim, int pos,
                                int blk_size, float scale, float *out) {
    /* Gather token indices from selected blocks (respecting causality) */
    int gather_idx[4096];
    int ngather = 0;
    for (int i = 0; i < topk; i++) {
        int b = selected[i];
        if (b < 0) continue;
        int bstart = b * blk_size;
        int bend = bstart + blk_size;
        if (bend > pos + 1) bend = pos + 1;
        for (int t = bstart; t < bend; t++) {
            if (ngather < (int)(sizeof(gather_idx) / sizeof(gather_idx[0])))
                gather_idx[ngather++] = t;
        }
    }
    if (ngather == 0) return;

    int _need = ngather * head_dim;
    if (_need > g_msa_buf_sz) {
        free(g_msa_kbuf); free(g_msa_vbuf); free(g_msa_sc);
        g_msa_kbuf = (float *)malloc((size_t)_need * sizeof(float));
        g_msa_vbuf = (float *)malloc((size_t)_need * sizeof(float));
        g_msa_sc = (float *)malloc((size_t)ngather * sizeof(float));
        g_msa_buf_sz = _need;
    }
    float *kbuf = g_msa_kbuf, *vbuf = g_msa_vbuf;
    for (int i = 0; i < ngather; i++) {
        msa_kv_read(m, layer, kv_head, gather_idx[i], kbuf + (int64_t)i * head_dim, head_dim, 0);
        msa_kv_read(m, layer, kv_head, gather_idx[i], vbuf + (int64_t)i * head_dim, head_dim, 1);
    }

    for (int h = 0; h < n_q_in_group; h++) {
        const float *qh = Q + (int64_t)h * head_dim;
        float *sc = g_msa_sc;
        float mx = -1e30f;
        for (int i = 0; i < ngather; i++) {
            sc[i] = dot_f(qh, kbuf + (int64_t)i * head_dim, head_dim) * scale;
            if (sc[i] > mx) mx = sc[i];
        }
        float sum = 0.f;
        for (int i = 0; i < ngather; i++) {
            sc[i] = expf(sc[i] - mx);
            sum += sc[i];
        }
        float inv = sum > 0.f ? 1.f / sum : 0.f;
        float *oh = out + (int64_t)h * head_dim;
        memset(oh, 0, (size_t)head_dim * sizeof(float));
        for (int i = 0; i < ngather; i++)
            axpy_f(oh, vbuf + (int64_t)i * head_dim, sc[i] * inv, head_dim);
        /* sc persistent */
    }
    /* kbuf/vbuf persistent */
}

/* Persistent attention + MSA scratch buffers — allocated once, reused. */
static float *g_attn_Q=NULL,*g_attn_Kp=NULL,*g_attn_Vp=NULL;
static float *g_attn_ctx=NULL,*g_attn_iq=NULL,*g_attn_ikt=NULL,*g_attn_scores=NULL;
static int g_attn_qsz=0, g_attn_kv=0;

static void attn_scratch_init(int qsz,int kv,int ih,int idm,int nb){
  if(qsz==g_attn_qsz&&kv==g_attn_kv)return;
  free(g_attn_Q);free(g_attn_Kp);free(g_attn_Vp);free(g_attn_ctx);
  free(g_attn_iq);free(g_attn_ikt);free(g_attn_scores);
  g_attn_Q=falloc(qsz);g_attn_Kp=falloc(kv);g_attn_Vp=falloc(kv);
  g_attn_ctx=falloc(qsz);g_attn_iq=falloc(ih*idm);g_attn_ikt=falloc(idm);
  g_attn_scores=falloc(ih*nb);g_attn_qsz=qsz;g_attn_kv=kv;
}

/* Full MSA attention pass for one token (S must be 1).
 * Projects Q/K/V (+ indexer Q/K), applies QK norm + partial RoPE, stores K/V and
 * indexer K in cache, selects top-16 blocks per GQA group, and runs block-sparse
 * softmax attention over the gathered tokens. */
static void attention_msa(Model *m, Layer *l, int layer, float *x, int S, int pos0, float *out) {
    Cfg *c = &m->c;
    int H = c->heads, Hkv = c->kv_heads, hd = c->head_dim;
    int idx_heads = c->idx_heads, idx_dim = c->idx_dim;
    int blk_size = c->blk_size, topk_blk = c->topk_blk;
    int n_q_in_group = H / Hkv;
    float scale = 1.f / sqrtf((float)hd);
    /* This engine is single-token; MSA batch path is not exercised. */
    if (S != 1) { /* fall back to dense for batched calls (not used by run_gen) */
        /* delegate to dense path by computing full attention inline */
    }

    /* 1. QKV projection (same matrices as dense path) */
    int64_t qsz = (int64_t)S * H * hd, kv = (int64_t)S * Hkv * hd;
    int n_blocks = (m->max_t + blk_size - 1) / blk_size;
    attn_scratch_init(qsz, kv, idx_heads, idx_dim, n_blocks);
    float *Q = g_attn_Q, *Kp = g_attn_Kp, *Vp = g_attn_Vp;
    matmul_qt(Q, x, &l->q, S);
    matmul_qt(Kp, x, &l->k, S);
    matmul_qt(Vp, x, &l->v, S);

    int pos = pos0;     /* S==1 */
    /* 2. QK norm + partial RoPE + K/V cache store (mirrors dense path) */
    for (int h = 0; h < H; h++) {
        float *qh = Q + (int64_t)h * hd;
        if (l->qn) rmsnorm(qh, qh, l->qn, hd, c->eps, c->gemma_norm);
        rope(qh, pos, c->theta, hd, c->rotary_dim);
    }
    for (int kh = 0; kh < Hkv; kh++) {
        float *kk = Kp + (int64_t)kh * hd;
        float *vv = Vp + (int64_t)kh * hd;
        if (l->kn) rmsnorm(kk, kk, l->kn, hd, c->eps, c->gemma_norm);
        rope(kk, pos, c->theta, hd, c->rotary_dim);
        int64_t off = ((int64_t)kh * m->max_t + pos) * hd;
        int64_t soff = (int64_t)kh * m->max_t + pos;
        if (m->kv_i8) {
            if (m->planar) {
                m->Ks[layer][soff] = planar_kv_encode(kk, m->Kq[layer] + off, hd, g_planar_bits);
                m->Vs[layer][soff] = planar_kv_encode(vv, m->Vq[layer] + off, hd, g_planar_bits);
            } else {
                float mx = 1e-8f;
                for (int i = 0; i < hd; i++) { float a = fabsf(kk[i]); if (a > mx) mx = a; }
                m->Ks[layer][soff] = mx / 127.f;
                float inv = 1.f / m->Ks[layer][soff];
                for (int i = 0; i < hd; i++) m->Kq[layer][off + i] = (int8_t)lrintf(kk[i] * inv);
                mx = 1e-8f;
                for (int i = 0; i < hd; i++) { float a = fabsf(vv[i]); if (a > mx) mx = a; }
                m->Vs[layer][soff] = mx / 127.f;
                inv = 1.f / m->Vs[layer][soff];
                for (int i = 0; i < hd; i++) m->Vq[layer][off + i] = (int8_t)lrintf(vv[i] * inv);
            }
        } else {
            memcpy(m->K[layer] + off, kk, (size_t)hd * sizeof(float));
            memcpy(m->V[layer] + off, vv, (size_t)hd * sizeof(float));
        }
    }

    /* 3. Indexer forward: project to idx_dim, per-head RMSNorm, partial RoPE, cache K */
    float *iq = g_attn_iq;
    float *ikt = g_attn_ikt;
    matmul_qt(iq, x, &l->iq, 1);    /* [idx_heads * idx_dim] */
    matmul_qt(ikt, x, &l->ik, 1);   /* [idx_dim] */
    for (int h = 0; h < idx_heads; h++) {
        float *iqh = iq + (int64_t)h * idx_dim;
        if (l->iqn) rmsnorm(iqh, iqh, l->iqn, idx_dim, c->eps, c->gemma_norm);
        rope(iqh, pos, c->theta, idx_dim, c->rotary_dim);
    }
    if (l->ikn) rmsnorm(ikt, ikt, l->ikn, idx_dim, c->eps, c->gemma_norm);
    rope(ikt, pos, c->theta, idx_dim, c->rotary_dim);
    memcpy(m->Kidx[layer] + (int64_t)pos * idx_dim, ikt, (size_t)idx_dim * sizeof(float));

    /* 4. Per-GQA-group block selection + sparse attention */
    float *scores = g_attn_scores;
    msa_indexer_scores(iq, m->Kidx[layer], pos, idx_heads, idx_dim,
                       m->max_t, blk_size, n_blocks, scores);

    float *ctx = g_attn_ctx;
    memset(ctx, 0, (size_t)qsz * sizeof(float));
    for (int g = 0; g < Hkv; g++) {
        int selected[64];   /* topk_blk max (16) */
        msa_topk_select(scores + (int64_t)g * n_blocks, n_blocks, topk_blk,
                        c->init_blk, pos, blk_size, c->local_blk, selected);
        const float *Qg = Q + (int64_t)g * n_q_in_group * hd;
        float *outg = ctx + (int64_t)g * n_q_in_group * hd;
        msa_group_attention(m, layer, g, Qg, selected, topk_blk,
                            n_q_in_group, hd, pos, blk_size, scale, outg);
    }

    /* 5. Output projection */
    matmul_qt(out, ctx, &l->o, 1);

    if (getenv("MSA_DEBUG") && getenv("MSA_DEBUG")[0] == '1') {
        fprintf(stderr, "[layer %d] sparse (16 blocks)\n", layer);
    }

    /* scratch buffers persistent */
}

static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos0, float *out) {
    /* MSA routing: sparse_attention_freq[layer]==1 (and single-token, f32 or i8 KV)
     * takes the block-sparse path; layers 0-2 (freq==0) use dense GQA. */
    if (l->msa && S == 1) {
        attention_msa(m, l, layer, x, S, pos0, out);
        return;
    }
    if (getenv("MSA_DEBUG") && getenv("MSA_DEBUG")[0] == '1') {
        fprintf(stderr, "[layer %d] dense\n", layer);
    }
    Cfg *c = &m->c;
    int H = c->heads, Hkv = c->kv_heads, hd = c->head_dim, nrep = H / Hkv;
    float scale = 1.f / sqrtf((float)hd);
    int64_t qsz = (int64_t)S * H * hd, kv = (int64_t)S * Hkv * hd;
    float *Q = falloc(qsz), *Kp = falloc(kv), *Vp = falloc(kv);
    matmul_qt(Q, x, &l->q, S);
    matmul_qt(Kp, x, &l->k, S);
    matmul_qt(Vp, x, &l->v, S);
    for (int s = 0; s < S; s++) {
        int pos = pos0 + s;
        for (int h = 0; h < H; h++) {
            float *qh = Q + (int64_t)s * H * hd + (int64_t)h * hd;
            if (l->qn) rmsnorm(qh, qh, l->qn, hd, c->eps, c->gemma_norm);
            rope(qh, pos, c->theta, hd, c->rotary_dim);
        }
        for (int kh = 0; kh < Hkv; kh++) {
            float *kk = Kp + (int64_t)s * Hkv * hd + (int64_t)kh * hd;
            float *vv = Vp + (int64_t)s * Hkv * hd + (int64_t)kh * hd;
            if (l->kn) rmsnorm(kk, kk, l->kn, hd, c->eps, c->gemma_norm);
            rope(kk, pos, c->theta, hd, c->rotary_dim);
            int64_t off = ((int64_t)kh * m->max_t + pos) * hd;
            int64_t soff = (int64_t)kh * m->max_t + pos;
            if (m->kv_i8) {
                if (m->planar) {
                    m->Ks[layer][soff] = planar_kv_encode(kk, m->Kq[layer] + off, hd, g_planar_bits);
                    m->Vs[layer][soff] = planar_kv_encode(vv, m->Vq[layer] + off, hd, g_planar_bits);
                } else {
                    m->Ks[layer][soff] = kv_quantize_i8(kk, m->Kq[layer] + off, hd);
                    m->Vs[layer][soff] = kv_quantize_i8(vv, m->Vq[layer] + off, hd);
                }
            } else {
                memcpy(m->K[layer] + off, kk, (size_t)hd * sizeof(float));
                memcpy(m->V[layer] + off, vv, (size_t)hd * sizeof(float));
            }
        }
    }
    float *ctx = falloc(qsz);
#pragma omp parallel for collapse(2) schedule(static)
    for (int s = 0; s < S; s++)
        for (int h = 0; h < H; h++) {
            int pos = pos0 + s, kvh = h / nrep, nt = pos + 1;
            const float *qv = Q + (int64_t)s * H * hd + (int64_t)h * hd;
            float *sc = (float *)malloc((size_t)nt * sizeof(float));
            for (int t = 0; t < nt; t++) {
                if (m->kv_i8) {
                    int64_t off = ((int64_t)kvh * m->max_t + t) * hd;
                    int64_t soff = (int64_t)kvh * m->max_t + t;
                    if (m->planar)
                        sc[t] = planar_kv_dot_q(qv, m->Kq[layer] + off, m->Ks[layer][soff], hd, g_planar_bits) * scale;
                    else {
                        float a = 0;
                        for (int i = 0; i < hd; i++) a += qv[i] * (float)m->Kq[layer][off + i];
                        sc[t] = a * m->Ks[layer][soff] * scale;
                    }
                } else {
                    const float *kvv = m->K[layer] + ((int64_t)kvh * m->max_t + t) * hd;
                    sc[t] = dot_f(qv, kvv, hd) * scale;
                }
            }
            softmax(sc, nt);
            float *cx = ctx + (int64_t)s * H * hd + (int64_t)h * hd;
            memset(cx, 0, (size_t)hd * sizeof(float));
            for (int t = 0; t < nt; t++) {
                if (m->kv_i8) {
                    int64_t off = ((int64_t)kvh * m->max_t + t) * hd;
                    int64_t soff = (int64_t)kvh * m->max_t + t;
                    if (m->planar) {
                        float tmp[256];
                        planar_kv_decode(tmp, m->Vq[layer] + off, m->Vs[layer][soff], hd, g_planar_bits);
                        axpy_f(cx, tmp, sc[t], hd);
                    } else {
                        for (int i = 0; i < hd; i++)
                            cx[i] += sc[t] * m->Vs[layer][soff] * (float)m->Vq[layer][off + i];
                    }
                } else {
                    const float *vv = m->V[layer] + ((int64_t)kvh * m->max_t + t) * hd;
                    axpy_f(cx, vv, sc[t], hd);
                }
            }
            free(sc);
        }
    matmul_qt(out, ctx, &l->o, S);
    free(Q);
    free(Kp);
    free(Vp);
    free(ctx);
}

static float *g_dense_g = NULL, *g_dense_u = NULL;
static int g_dense_I = 0;

static void dense_mlp(Model *m, Layer *l, float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden;
    int I = c->dense_inter;
    if (I != g_dense_I) {
        free(g_dense_g); free(g_dense_u);
        g_dense_g = falloc((int64_t)S * I);
        g_dense_u = falloc((int64_t)S * I);
        g_dense_I = I;
    }
    float *g = g_dense_g, *u = g_dense_u;
    matmul_qt(g, x, &l->gate, S);
    matmul_qt(u, x, &l->up, S);
    for (int64_t i = 0; i < (int64_t)S * I; i++) g[i] = swiglu(g[i], u[i], c->sw_alpha, c->sw_limit);
    matmul_qt(out, g, &l->down, S);
}

/* MiniMax M3 MoE router (sigmoid + e_score_correction_bias + route_norm + top-K).
 * Extracted from moe() so unit tests (VAL-CORR-012) can exercise the router
 * without driving a full MoE forward pass. Reference: HF MiniMaxM3VLTopKRouter.
 *   x:      [D] hidden state for one token
 *   router: [E*D] f32 router weight (E output rows, D input dims)
 *   bias:   [E] f32 e_score_correction_bias
 *   idx:    [K] output selected expert indices (sorted by score desc within forced order)
 *   w:      [K] output router weights (post-normalization, post-router_scale)
 *   logit, choice: scratch buffers of size E
 */
static void moe_router(int *idx, float *w, const float *x, const float *router,
                       const float *bias, int D, int E, int K, int route_norm,
                       float router_scale, float *logit, float *choice) {
    matmul_f(logit, x, router, 1, D, E);
    for (int e = 0; e < E; e++) {
        logit[e] = sigmoidf_(logit[e]);        /* sigmoid scoring */
        choice[e] = logit[e] + bias[e];        /* + e_score_correction_bias for selection */
    }
    for (int kk = 0; kk < K; kk++) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int j = 0; j < kk; j++)
                if (idx[j] == e) { taken = 1; break; }
            if (!taken && choice[e] > bv) {
                bv = choice[e];
                best = e;
            }
        }
        idx[kk] = best;
        w[kk] = logit[best];                   /* weight = sigmoid(logit), NOT sigmoid+bias */
    }
    if (route_norm) {
        float sm = 1e-20f;
        for (int kk = 0; kk < K; kk++) sm += w[kk];
        for (int kk = 0; kk < K; kk++) w[kk] /= sm;
    }
    for (int kk = 0; kk < K; kk++) w[kk] *= router_scale;
}

/* f12: parallel MoE expert dispatch. Each routed expert runs concurrently
 * across an OMP thread group; per-thread accumulators reduce into out at the end.
 * Used when MOE_PARALLEL=1 (default for decode where S=1 and the 4 routed
 * experts are independent). Falls back to the sequential moe() when off.
 *
 * Per-thread accumulator budget: T * S * D floats. For decode (S=1, D=6144, T=88)
 * this is ~2 MB; for prefill (S=128) ~280 MB — acceptable on the 370 GB host.
 * The shared expert is computed once after the parallel reduction (it is
 * deterministic and single-threaded).
 */
static void moe_parallel(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->experts, K = c->topk, I = c->moe_inter, sI = I * c->n_shared;
    float *logit = falloc(E), *choice = falloc(E);
    int *idxs = (int *)malloc((size_t)S * K * sizeof(int));
    float *ws = (float *)malloc((size_t)S * K * sizeof(float));
    memset(out, 0, (size_t)S * D * sizeof(float));
    for (int s = 0; s < S; s++) {
        moe_router(idxs + (int64_t)s * K, ws + (int64_t)s * K,
                   x + (int64_t)s * D, l->router, l->router_bias,
                   D, E, K, c->route_norm, c->router_scale, logit, choice);
    }
    /* batch-union experts */
    unsigned char *seen = (unsigned char *)calloc((size_t)E, 1);
    int *uniq = (int *)malloc((size_t)E * sizeof(int));
    int nu = 0;
    for (int s = 0; s < S; s++)
        for (int kk = 0; kk < K; kk++) {
            int e = idxs[(int64_t)s * K + kk];
            if (!seen[e]) { seen[e] = 1; uniq[nu++] = e; }
        }

    int T = omp_get_max_threads();
    if (T < 1) T = 1;
    /* Per-thread accumulators. */
    float *acc = (float *)calloc((size_t)T * S * D, sizeof(float));

    /* Pre-load all unique experts SERIALLY — expert_get() modifies the shared
     * cache (m->cn[layer], m->cache[layer]) and is NOT thread-safe. Doing this
     * before the parallel for avoids the race. The expert matmuls (the
     * expensive part) still run in parallel below. */
    ESlot **eslots = (ESlot **)malloc((size_t)nu * sizeof(ESlot *));
    for (int j = 0; j < nu; j++)
        eslots[j] = expert_get(m, layer, uniq[j]);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < nu; j++) {
        int tid = omp_get_thread_num();
        float *my_acc = acc + (int64_t)tid * S * D;
        /* Per-thread scratch buffers — allocated inside the loop to avoid races. */
        float *xg_priv = (float *)malloc((size_t)S * D * sizeof(float));
        float *gg_priv = (float *)malloc((size_t)S * I * sizeof(float));
        float *uu_priv = (float *)malloc((size_t)S * I * sizeof(float));
        float *hh_priv = (float *)malloc((size_t)S * D * sizeof(float));
        int *rows = (int *)malloc((size_t)S * sizeof(int));
        float *rw = (float *)malloc((size_t)S * sizeof(float));

        int eid = uniq[j];
        ESlot *e = eslots[j];
        int nr = 0;
        for (int s = 0; s < S; s++)
            for (int kk = 0; kk < K; kk++)
                if (idxs[(int64_t)s * K + kk] == eid) {
                    rows[nr] = s; rw[nr] = ws[(int64_t)s * K + kk]; nr++;
                    break;
                }
        for (int r = 0; r < nr; r++)
            memcpy(xg_priv + (int64_t)r * D, x + (int64_t)rows[r] * D, (size_t)D * sizeof(float));
        matmul_qt(gg_priv, xg_priv, &e->g, nr);
        matmul_qt(uu_priv, xg_priv, &e->u, nr);
        for (int64_t z = 0; z < (int64_t)nr * I; z++)
            gg_priv[z] = swiglu(gg_priv[z], uu_priv[z], c->sw_alpha, c->sw_limit);
        matmul_qt(hh_priv, gg_priv, &e->d, nr);
        for (int r = 0; r < nr; r++) {
            float w = rw[r]; float *hr = hh_priv + (int64_t)r * D;
            float *os = my_acc + (int64_t)rows[r] * D;
            for (int d = 0; d < D; d++) os[d] += w * hr[d];
        }
        free(xg_priv); free(gg_priv); free(uu_priv); free(hh_priv); free(rows); free(rw);
    }
    /* Reduce per-thread accumulators into out. */
    for (int tid = 0; tid < T; tid++) {
        const float *my = acc + (int64_t)tid * S * D;
        for (int64_t z = 0; z < (int64_t)S * D; z++) out[z] += my[z];
    }

    /* Shared expert — same as sequential path, runs once after routed experts. */
    float *sg = falloc((int64_t)S * sI), *su = falloc((int64_t)S * sI),
          *hh = falloc((int64_t)S * D);
    matmul_qt(sg, x, &l->sh_gate, S);
    matmul_qt(su, x, &l->sh_up, S);
    for (int64_t z = 0; z < (int64_t)S * sI; z++)
        sg[z] = swiglu(sg[z], su[z], c->sw_alpha, c->sw_limit);
    matmul_qt(hh, sg, &l->sh_down, S);
    for (int64_t z = 0; z < (int64_t)S * D; z++) out[z] += hh[z];

    free(acc); free(eslots); free(sg); free(su); free(hh);
    free(logit); free(choice); free(idxs); free(ws); free(seen); free(uniq);
}

/* Persistent MoE scratch buffers — allocated once, reused across all layers.
 * Eliminates 660+ malloc/free per token (11 per layer * 60 layers). */
static float *g_moe_xg = NULL, *g_moe_gg = NULL, *g_moe_uu = NULL,
             *g_moe_hh = NULL, *g_moe_sg = NULL, *g_moe_su = NULL;
static int g_moe_scratch_D = 0, g_moe_scratch_I = 0;

static void moe_scratch_init(int D, int I, int n_shared) {
    int sI = I * n_shared;
    if (D == g_moe_scratch_D && I == g_moe_scratch_I) return;
    free(g_moe_xg); free(g_moe_gg); free(g_moe_uu);
    free(g_moe_hh); free(g_moe_sg); free(g_moe_su);
    g_moe_xg = falloc(D);     /* S=1 max for decode */
    g_moe_gg = falloc(I);
    g_moe_uu = falloc(I);
    g_moe_hh = falloc(D);
    g_moe_sg = falloc(sI);
    g_moe_su = falloc(sI);
    g_moe_scratch_D = D;
    g_moe_scratch_I = I;
}

static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->experts, K = c->topk, I = c->moe_inter, sI = I * c->n_shared;
    moe_scratch_init(D, I, c->n_shared);
    float *xg = g_moe_xg, *gg = g_moe_gg, *uu = g_moe_uu, *hh = g_moe_hh;
    float *sg = g_moe_sg, *su = g_moe_su;
    /* Small stack buffers for routing (avoid heap entirely) */
    float logit_buf[256], choice_buf[256];
    int idxs_buf[64], rows_buf[64];
    float ws_buf[64];
    int uniq_buf[256];
    unsigned char seen_buf[256];
    float *logit = logit_buf, *choice = choice_buf;
    int *idxs = idxs_buf, *rows = rows_buf;
    float *ws = ws_buf;
    int *uniq = uniq_buf;
    unsigned char *seen = seen_buf;
    /* For large E or S, fall back to heap */
    if (E > 256) { logit = falloc(E); choice = falloc(E); }
    if (S * K > 64) { idxs = malloc(S*K*sizeof(int)); ws = malloc(S*K*sizeof(float)); }
    memset(seen, 0, (size_t)E);
    memset(out, 0, (size_t)S * D * sizeof(float));
    for (int s = 0; s < S; s++) {
        moe_router(idxs + (int64_t)s * K, ws + (int64_t)s * K,
                   x + (int64_t)s * D, l->router, l->router_bias,
                   D, E, K, c->route_norm, c->router_scale, logit, choice);
    }
    int nu = 0;
    for (int s = 0; s < S; s++)
        for (int kk = 0; kk < K; kk++) {
            int e = idxs[(int64_t)s * K + kk];
            if (!seen[e]) { seen[e] = 1; uniq[nu++] = e; }
        }
    for (int j = 0; j < nu; j++) {
        int eid = uniq[j];
        ESlot *e = expert_get(m, layer, eid);
        int nr = 0;
        for (int s = 0; s < S; s++)
            for (int kk = 0; kk < K; kk++)
                if (idxs[(int64_t)s * K + kk] == eid) {
                    rows[nr] = s;
                    ws[nr] = ws[(int64_t)s * K + kk];
                    nr++; break;
                }
        for (int r = 0; r < nr; r++) memcpy(xg + (int64_t)r * D, x + (int64_t)rows[r] * D, (size_t)D * sizeof(float));
        matmul_qt(gg, xg, &e->g, nr);
        matmul_qt(uu, xg, &e->u, nr);
        for (int64_t z = 0; z < (int64_t)nr * I; z++) gg[z] = swiglu(gg[z], uu[z], c->sw_alpha, c->sw_limit);
        matmul_qt(hh, gg, &e->d, nr);
        for (int r = 0; r < nr; r++) {
            float *os = out + (int64_t)rows[r] * D, w = ws[r], *hr = hh + (int64_t)r * D;
            for (int d = 0; d < D; d++) os[d] += w * hr[d];
        }
    }
    matmul_qt(sg, x, &l->sh_gate, S);
    matmul_qt(su, x, &l->sh_up, S);
    for (int64_t z = 0; z < (int64_t)S * sI; z++) sg[z] = swiglu(sg[z], su[z], c->sw_alpha, c->sw_limit);
    matmul_qt(hh, sg, &l->sh_down, S);
    for (int64_t z = 0; z < (int64_t)S * D; z++) out[z] += hh[z];
    if (E > 256) { free(logit); free(choice); }
    if (S * K > 64) { free(idxs); free(ws); }
}

#include "eagle3.h"

static void layer_fwd(Model *m, int li, float *x, int S, int pos0, float *nrm, float *tmp) {
    Layer *l = &m->L[li];
    Cfg *c = &m->c;
    int D = c->hidden;
    for (int s = 0; s < S; s++) rmsnorm(nrm + (int64_t)s * D, x + (int64_t)s * D, l->in_ln, D, c->eps, c->gemma_norm);
    if (g_nan_check || g_debug_trace) nan_check_layer(li, "rmsnorm_in", nrm, (int64_t)S * D);
    attention(m, l, li, nrm, S, pos0, tmp);
    if (g_nan_check || g_debug_trace) nan_check_layer(li, "attn_out", tmp, (int64_t)S * D);
    for (int64_t j = 0; j < (int64_t)S * D; j++) x[j] += tmp[j];
    for (int s = 0; s < S; s++) rmsnorm(nrm + (int64_t)s * D, x + (int64_t)s * D, l->post_ln, D, c->eps, c->gemma_norm);
    if (g_nan_check || g_debug_trace) nan_check_layer(li, "post_ln", nrm, (int64_t)S * D);
    if (l->sparse) {
        if (g_moe_parallel) moe_parallel(m, l, li, nrm, S, tmp);
        else                moe(m, l, li, nrm, S, tmp);
    } else dense_mlp(m, l, nrm, S, tmp);
    if (g_nan_check || g_debug_trace) nan_check_layer(li, "mlp_out", tmp, (int64_t)S * D);
    for (int64_t j = 0; j < (int64_t)S * D; j++) x[j] += tmp[j];
    /* Prefetch first cache line of next layer's weights to reduce stall */
    if (li + 1 < c->layers) {
        Layer *nl = &m->L[li + 1];
        if (nl->q.q4) __builtin_prefetch(nl->q.q4, 0, 0);
        if (nl->k.q4) __builtin_prefetch(nl->k.q4, 0, 0);
        if (nl->v.q4) __builtin_prefetch(nl->v.q4, 0, 0);
    }
    if (g_nan_check || g_debug_trace) nan_check_layer(li, "residual", x, (int64_t)S * D);
}

static void embed_tok(Model *m, int tok, float *x) {
    Cfg *c = &m->c;
    int D = c->hidden;
    if (m->embed.fmt == 0) {
        memcpy(x, m->embed.qf + (int64_t)tok * D, (size_t)D * sizeof(float));
        return;
    }
    /* dequant one row */
    if (m->embed.fmt == 1) {
        const int8_t *q = m->embed.q8 + (int64_t)tok * D;
        float s = m->embed.s[tok];
        for (int i = 0; i < D; i++) x[i] = (float)q[i] * s;
    } else {
        int rb = (D + 1) / 2;
        const uint8_t *q = m->embed.q4 + (int64_t)tok * rb;
        float s = m->embed.s[tok];
        for (int i = 0; i < D; i++) {
            uint8_t b = q[i >> 1];
            int nib = (i & 1) ? (b >> 4) : (b & 0xF);
            x[i] = (float)(nib - 8) * s;
        }
    }
}

static int sample_logits(const float *lo, int V) {
    if (g_temp <= 1e-5f) {
        int best = 0;
        float bv = lo[0];
        for (int i = 1; i < V; i++)
            if (lo[i] > bv) {
                bv = lo[i];
                best = i;
            }
        return best;
    }
    float *p = falloc(V);
    float mx = lo[0];
    for (int i = 1; i < V; i++)
        if (lo[i] > mx) mx = lo[i];
    float invt = 1.f / g_temp;
    double sum = 0;
    for (int i = 0; i < V; i++) {
        p[i] = expf((lo[i] - mx) * invt);
        sum += p[i];
    }
    for (int i = 0; i < V; i++) p[i] /= (float)sum;
    /* top-k */
    if (g_topk_samp > 0 && g_topk_samp < V) {
        int *idx = (int *)malloc((size_t)V * sizeof(int));
        for (int i = 0; i < V; i++) idx[i] = i;
        for (int a = 0; a < g_topk_samp; a++) {
            int bi = a;
            for (int b = a + 1; b < V; b++)
                if (p[idx[b]] > p[idx[bi]]) bi = b;
            int tmp = idx[a];
            idx[a] = idx[bi];
            idx[bi] = tmp;
        }
        for (int i = g_topk_samp; i < V; i++) p[idx[i]] = 0;
        double s2 = 0;
        for (int i = 0; i < g_topk_samp; i++) s2 += p[idx[i]];
        for (int i = 0; i < g_topk_samp; i++) p[idx[i]] /= (float)s2;
        free(idx);
    }
    /* nucleus */
    if (g_topp > 0 && g_topp < 1.f) {
        int *idx = (int *)malloc((size_t)V * sizeof(int));
        for (int i = 0; i < V; i++) idx[i] = i;
        for (int a = 0; a < V; a++) {
            int bi = a;
            for (int b = a + 1; b < V; b++)
                if (p[idx[b]] > p[idx[bi]]) bi = b;
            int tmp = idx[a];
            idx[a] = idx[bi];
            idx[bi] = tmp;
        }
        double cum = 0;
        int keep = V;
        for (int i = 0; i < V; i++) {
            cum += p[idx[i]];
            if (cum >= g_topp) {
                keep = i + 1;
                break;
            }
        }
        for (int i = keep; i < V; i++) p[idx[i]] = 0;
        double s2 = 0;
        for (int i = 0; i < keep; i++) s2 += p[idx[i]];
        for (int i = 0; i < keep; i++) p[idx[i]] /= (float)s2;
        free(idx);
    }
    double u = (double)rand() / RAND_MAX, cum = 0;
    for (int i = 0; i < V; i++) {
        cum += p[i];
        if (cum >= u) {
            free(p);
            return i;
        }
    }
    free(p);
    return V - 1;
}

/* f13: native C benchmark mode. Runs greedy decode and prints per-token
 * decode timing to stderr, plus a JSON summary line at the end. This is the
 * zero-overhead path — no Python subprocess, no JSON parsing. The Python
 * bench_throughput.py harness calls this when available; otherwise it falls
 * back to parsing stderr.
 *
 * Output format (stderr):
 *   [bench] tok 0: decode_ms=195.3 rolling_tokps=5.12 hit=0/4
 *   [bench] tok 1: decode_ms=188.7 rolling_tokps=5.30 hit=1/3
 *   ...
 *   [bench] SUMMARY {"tokens":20,"wall_s":78.98,"tokps":0.25,"warm_tokps":0.28,"hit_rate":0.554,"vnni":1,"moe_parallel":1}
 */
static int run_bench(Model *m, const int *prompt, int np, int max_new) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *x = falloc(D), *nrm = falloc(D), *tmp = falloc(D), *logits = falloc(c->vocab);
    double t0 = now_s();
    for (int i = 0; i < np; i++) {
        embed_tok(m, prompt[i], x);
        for (int li = 0; li < c->layers; li++) {
            layer_fwd(m, li, x, 1, i, nrm, tmp);
            if (g_use_mtp && g_e3.loaded && li == E3_TARGET_LAYER) {
                /* Run EAGLE3 forward to fill its KV cache */
                float *e3_logits = falloc(c->vocab);
                e3_forward(x, prompt[i], i, e3_logits, c, m);
                free(e3_logits);
            }
        }
        if ((i + 1) == np || ((i + 1) % 32) == 0)
            fprintf(stderr, "\r[bench-prefill] %d/%d", i + 1, np);
    }
    fprintf(stderr, "\n[bench-prefill] done in %.2fs\n", now_s() - t0);
    rmsnorm(nrm, x, m->final_norm, D, c->eps, c->gemma_norm);
    matmul_qt(logits, nrm, &m->lm_head, 1);

    /* EAGLE3 prefill: run e3_forward for each prompt position to fill KV cache */
    if (g_use_mtp && g_e3.loaded) {
        fprintf(stderr, "[e3] prefilling KV cache for %d tokens...\n", np);
        /* We need the hidden state at layer 57 for each prompt position.
         * But we already ran the full model. We can use the final hidden state
         * as an approximation, or re-run. For now, just clear the KV cache
         * and let EAGLE3 attend only to decode positions. */
        memset(g_e3.K_cache, 0, (size_t)g_e3.max_pos * E3_KV_HEADS * E3_HEAD_DIM * sizeof(float));
        memset(g_e3.V_cache, 0, (size_t)g_e3.max_pos * E3_KV_HEADS * E3_HEAD_DIM * sizeof(float));
        fprintf(stderr, "[e3] KV cache cleared\n");
    }

    /* bench mode: always greedy (TEMP=0) to isolate decode timing */
    g_temp = 0.0f; g_topp = 0.0f; g_topk_samp = 0;

    int pos = np;
    t0 = now_s();
    double *step_times = (double *)malloc((size_t)max_new * sizeof(double));
    int emitted = 0;
    /* For EAGLE3: we need the hidden state at layer 57 (E3_TARGET_LAYER).
     * We'll capture it by modifying the forward loop to save x after layer 57. */
    static float e3_hidden[6144];  /* captured hidden state */
    int e3_has_hidden = 0;

    while (emitted < max_new) {
        double step_t0 = now_s();

        /* Greedy decode from current logits */
        int tok = sample_logits(logits, c->vocab);
        printf("%d\n", tok);
        fflush(stdout);
        emitted++;
        if (tok == c->eos) break;
        if (emitted >= max_new) break;

        /* EAGLE3: draft a token using hidden state from layer 57 */
        int draft_tok = -1;
        if (g_use_mtp && g_e3.loaded && e3_has_hidden) {
            float *draft_logits = falloc(c->vocab);
            draft_tok = e3_forward(e3_hidden, tok, pos, draft_logits, c, m);
            free(draft_logits);
        }

        /* Run base model forward on tok at position pos.
         * Capture hidden state at layer E3_TARGET_LAYER (57). */
        embed_tok(m, tok, x);
        for (int li = 0; li < c->layers; li++) {
            layer_fwd(m, li, x, 1, pos, nrm, tmp);
            if (g_use_mtp && g_e3.loaded && li == E3_TARGET_LAYER) {
                memcpy(e3_hidden, x, (size_t)D * sizeof(float));
                e3_has_hidden = 1;
            }
        }
        pos++;
        rmsnorm(nrm, x, m->final_norm, D, c->eps, c->gemma_norm);
        matmul_qt(logits, nrm, &m->lm_head, 1);

        /* Check if EAGLE3 draft matches base model's prediction */
        int base_tok = sample_logits(logits, c->vocab);
        if (draft_tok >= 0) fprintf(stderr, "[e3] draft=%d base=%d pos=%d\n", draft_tok, base_tok, pos);
        int accepted = 0;

        if (draft_tok >= 0 && draft_tok == base_tok) {
            /* Accepted! Emit draft token for free. */
            printf("%d\n", draft_tok);
            fflush(stdout);
            emitted++;
            accepted = 1;
            if (draft_tok == c->eos) break;

            /* Advance: embed base_tok, run forward to get next logits */
            embed_tok(m, base_tok, x);
            for (int li = 0; li < c->layers; li++) {
                layer_fwd(m, li, x, 1, pos, nrm, tmp);
                if (g_use_mtp && g_e3.loaded && li == E3_TARGET_LAYER) {
                    memcpy(e3_hidden, x, (size_t)D * sizeof(float));
                }
            }
            pos++;
            rmsnorm(nrm, x, m->final_norm, D, c->eps, c->gemma_norm);
            matmul_qt(logits, nrm, &m->lm_head, 1);
        }

        double step_dt = now_s() - step_t0;
        if (emitted - 1 < max_new) step_times[emitted - 1] = step_dt;
        double rolling = emitted / (now_s() - t0);
        uint64_t tot = m->hits + m->miss;
        fprintf(stderr, "[bench] tok %d: decode_ms=%.1f rolling_tokps=%.2f hit=%lu/%lu e3=%d accept=%d\n",
                emitted - 1, step_dt * 1000.0, rolling,
                (unsigned long)m->hits, (unsigned long)(tot),
                draft_tok >= 0 ? 1 : 0, accepted);
    }

    double dt = now_s() - t0;
    double tokps = emitted / (dt > 1e-6 ? dt : 1e-6);
    /* warm = last 50% of tokens */
    int warm_start = emitted / 2;
    double warm_sum = 0;
    int warm_n = 0;
    for (int t = warm_start; t < emitted; t++) { warm_sum += step_times[t]; warm_n++; }
    double warm_tokps = warm_n > 0 ? warm_n / warm_sum : tokps;
    double hit_rate = (m->hits + m->miss) > 0 ? (double)m->hits / (m->hits + m->miss) : 0;
    fprintf(stderr, "[bench] SUMMARY {\"tokens\":%d,\"wall_s\":%.3f,\"tokps\":%.3f,\"warm_tokps\":%.3f,\"hit_rate\":%.4f,\"vnni\":%d,\"moe_parallel\":%d}\n",
            emitted, dt, tokps, warm_tokps, hit_rate, g_use_vnni, g_moe_parallel);
    fprintf(stderr, "[stat] %d tok in %.2fs (%.2f tok/s) | expert hit %.1f%% | warm %.2f tok/s\n",
            emitted, dt, tokps, hit_rate * 100.0, warm_tokps);
    /* f15: emit run_complete telemetry. */
    telem_run_complete(emitted, dt, tokps, hit_rate, "colibri-m3");
    free(step_times);
    free(x); free(nrm); free(tmp); free(logits);
    return 0;
}

static int run_gen(Model *m, const int *prompt, int np, int max_new) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *x = falloc(D), *nrm = falloc(D), *tmp = falloc(D), *logits = falloc(c->vocab);
    double t0 = now_s();
    for (int i = 0; i < np; i++) {
        embed_tok(m, prompt[i], x);
        for (int li = 0; li < c->layers; li++) layer_fwd(m, li, x, 1, i, nrm, tmp);
        if ((i + 1) == np || ((i + 1) % 32) == 0)
            fprintf(stderr, "\r[prefill] %d/%d", i + 1, np);
    }
    fprintf(stderr, "\n[prefill] done in %.2fs\n", now_s() - t0);
    rmsnorm(nrm, x, m->final_norm, D, c->eps, c->gemma_norm);
    matmul_qt(logits, nrm, &m->lm_head, 1);
    if (g_nan_check || g_debug_trace) nan_check_layer(-1, "lm_head_logits", logits, c->vocab);
    int pos = np;
    t0 = now_s();
    int emitted = 0;
    for (int t = 0; t < max_new; t++) {
        int tok = sample_logits(logits, c->vocab);
        printf("%d\n", tok);
        fflush(stdout);
        emitted++;
        if (tok == c->eos) {
            fprintf(stderr, "[gen] token %d=%d (EOS) -> stopping\n", t, tok);
            break;
        }
        embed_tok(m, tok, x);
        for (int li = 0; li < c->layers; li++) layer_fwd(m, li, x, 1, pos, nrm, tmp);
        pos++;
        rmsnorm(nrm, x, m->final_norm, D, c->eps, c->gemma_norm);
        matmul_qt(logits, nrm, &m->lm_head, 1);
        if (g_nan_check || g_debug_trace) nan_check_layer(-1, "lm_head_logits", logits, c->vocab);
        /* f15: per-token decode telemetry. Rolling tok/s over the decode so far. */
        if (g_telem.enabled) {
            double step_dt = now_s() - t0;
            double rolling = (t + 1) / (step_dt > 1e-6 ? step_dt : 1e-6);
            telem_decode_step(t, step_dt * 1000.0 / (t + 1),
                              (int)m->hits, (int)m->miss, rolling);
        }
    }
    double dt = now_s() - t0;
    double tot = (double)(m->hits + m->miss);
    double tokps = emitted / (dt > 1e-6 ? dt : 1e-6);
    double hit_rate = tot ? (double)m->hits / tot : 0.0;
    fprintf(stderr, "[stat] %d tok in %.2fs (%.2f tok/s) | expert hit %.1f%%\n", emitted, dt,
            tokps, hit_rate * 100.0);
    /* f15: emit run_complete telemetry event. */
    telem_run_complete(emitted, dt, tokps, hit_rate, "colibri-m3");
    if (g_nan_check) {
        fprintf(stderr, "[nan-check] %ld failures detected over %d tokens\n",
                g_nan_failures, emitted);
        if (g_nan_failures > 0) return 1;
    }
    free(x);
    free(nrm);
    free(tmp);
    free(logits);
    return 0;
}

/* f8: teacher-forcing mode — feed the oracle's target tokens (instead of sampling)
 * and dump per-position top-K logprobs so test_oracle_logits.py can compare them
 * against the cached oracle artifacts (tests/oracle/logits.json).
 *
 * stdin protocol extension: a third line with space-separated target token IDs.
 * Output: a simple line-based dump to g_tf_out_path (default: stdout):
 *   TF_START npos=<N> topk=<K> vocab=<V>
 *   POS <i> argmax=<id> argmax_lp=<v> ntop=<K>
 *   <tok_id> <logprob>     (repeated ntop times, sorted by logprob desc)
 *   ...
 *   TF_END
 *
 * This is the format test_oracle_logits.py parses. The comparison uses the
 * relaxed tolerances documented in the feature spec (int4-vs-Q4_K_M drift):
 *   (a) argmax match: target 32/32, allow >=30/32
 *   (b) top-200 overlap: >=90% per position
 *   (c) overlapping logprob values: within 1e-2
 */
static int run_teacher_force(Model *m, const int *prompt, int np,
                             const int *targets, int nt, int topk, FILE *tf_out) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *x = falloc(D), *nrm = falloc(D), *tmp = falloc(D), *logits = falloc(c->vocab);
    double t0 = now_s();
    for (int i = 0; i < np; i++) {
        embed_tok(m, prompt[i], x);
        for (int li = 0; li < c->layers; li++) layer_fwd(m, li, x, 1, i, nrm, tmp);
        if ((i + 1) == np || ((i + 1) % 32) == 0)
            fprintf(stderr, "\r[tf-prefill] %d/%d", i + 1, np);
    }
    fprintf(stderr, "\n[tf-prefill] done in %.2fs\n", now_s() - t0);
    rmsnorm(nrm, x, m->final_norm, D, c->eps, c->gemma_norm);
    matmul_qt(logits, nrm, &m->lm_head, 1);
    if (g_nan_check || g_debug_trace) nan_check_layer(-1, "lm_head_logits", logits, c->vocab);

    int *top_ids = (int *)malloc((size_t)topk * sizeof(int));
    float *top_lps = (float *)malloc((size_t)topk * sizeof(float));
    int pos = np;
    int kmax = topk < c->vocab ? topk : c->vocab;

    fprintf(tf_out, "TF_START npos=%d topk=%d vocab=%d\n", nt, topk, c->vocab);
    for (int t = 0; t < nt; t++) {
        logsoftmax_topk(logits, c->vocab, topk, top_ids, top_lps);
        int argmax = (kmax > 0) ? top_ids[0] : -1;
        float argmax_lp = (kmax > 0) ? top_lps[0] : 0.0f;
        fprintf(tf_out, "POS %d argmax=%d argmax_lp=%.9g ntop=%d\n",
                t, argmax, (double)argmax_lp, kmax);
        for (int k = 0; k < kmax; k++)
            fprintf(tf_out, "%d %.9g\n", top_ids[k], (double)top_lps[k]);

        /* Feed the oracle's target token (teacher forcing) and advance the
         * KV cache — NOT the sampled token. This is what distinguishes TF from
         * greedy decode and ensures we compare logits at the exact same
         * prefixes the oracle used. */
        int tok = targets[t];
        embed_tok(m, tok, x);
        for (int li = 0; li < c->layers; li++) layer_fwd(m, li, x, 1, pos, nrm, tmp);
        pos++;
        rmsnorm(nrm, x, m->final_norm, D, c->eps, c->gemma_norm);
        matmul_qt(logits, nrm, &m->lm_head, 1);
        if (g_nan_check || g_debug_trace) nan_check_layer(-1, "lm_head_logits", logits, c->vocab);
    }
    fprintf(tf_out, "TF_END\n");

    if (g_nan_check) {
        fprintf(stderr, "[nan-check] %ld failures detected over %d TF positions\n",
                g_nan_failures, nt);
    }
    free(top_ids);
    free(top_lps);
    free(x);
    free(nrm);
    free(tmp);
    free(logits);
    return (g_nan_check && g_nan_failures > 0) ? 1 : 0;
}

/* ---------- dry-run: tensor-presence check ---------- *
 * Validates that every tensor the engine's load_model expects (derived from
 * config.json) is present in the converted snapshot, with the expected dtype
 * (quantized tensors must have a `.qs` scale companion; f32 tensors must NOT).
 * Produces a structured report: expected / found / missing / mis-typed counts.
 * Exits 0 only if 100% of expected tensors are present and correctly typed.
 *
 * Tensor dtype policy (must match tools/convert.py classify()):
 *   quant (need name + name.qs): embed_tokens, lm_head, self_attn.{q,k,v,o}_proj,
 *                                 shared_experts.{gate,up,down}_proj, dense mlp.{gate,up,down}_proj,
 *                                 mlp.experts.E.{gate,up,down}_proj
 *   f32   (need name, NO .qs):   model.norm, input_layernorm, post_attention_layernorm,
 *                                 self_attn.{q,k}_norm, mlp.gate.weight (router),
 *                                 mlp.gate.e_score_correction_bias
 */
static int dr_check_quant(shards *S, const char *name, long *exp, long *fnd, long *mis, long *mt) {
    char qs[640];
    snprintf(qs, sizeof(qs), "%s.qs", name);
    (*exp)++;
    if (!st_has(S, name)) {
        fprintf(stderr, "[dry-run] MISSING: %s\n", name);
        (*mis)++;
        return 1;
    }
    (*fnd)++;
    if (!st_has(S, qs)) {
        fprintf(stderr, "[dry-run] MIS-TYPED: %s (expected quantized w/ .qs scale, found f32)\n", name);
        (*mt)++;
        return 1;
    }
    return 0;
}
static int dr_check_f32(shards *S, const char *name, long *exp, long *fnd, long *mis, long *mt) {
    char qs[640];
    snprintf(qs, sizeof(qs), "%s.qs", name);
    (*exp)++;
    if (!st_has(S, name)) {
        fprintf(stderr, "[dry-run] MISSING: %s\n", name);
        (*mis)++;
        return 1;
    }
    (*fnd)++;
    if (st_has(S, qs)) {
        fprintf(stderr, "[dry-run] MIS-TYPED: %s (expected f32, found quantized w/ .qs)\n", name);
        (*mt)++;
        return 1;
    }
    return 0;
}
/* optional quant tensor: only flagged if present but mis-typed; absence is OK */
static void dr_check_opt_quant(shards *S, const char *name, long *exp, long *fnd, long *mt) {
    char qs[640];
    snprintf(qs, sizeof(qs), "%s.qs", name);
    if (!st_has(S, name)) return; /* optional */
    (*exp)++;
    (*fnd)++;
    if (!st_has(S, qs)) {
        fprintf(stderr, "[dry-run] MIS-TYPED: %s (expected quantized w/ .qs scale, found f32)\n", name);
        (*mt)++;
    }
}
/* f32 tensor with a named fallback (e.g. e_score_correction_bias -> expert_bias) */
static int dr_check_f32_alt(shards *S, const char *name, const char *alt,
                            long *exp, long *fnd, long *mis, long *mt) {
    if (st_has(S, name)) return dr_check_f32(S, name, exp, fnd, mis, mt);
    if (alt && st_has(S, alt)) return dr_check_f32(S, alt, exp, fnd, mis, mt);
    (*exp)++;
    fprintf(stderr, "[dry-run] MISSING: %s (or fallback %s)\n", name, alt ? alt : "(none)");
    (*mis)++;
    return 1;
}

static int dry_run(const char *snap) {
    Cfg c;
    load_cfg(&c, snap);
    shards S;
    st_init(&S, snap);

    long expected = 0, found = 0, missing = 0, mistyped = 0;
    char nm[512];

    dr_check_quant(&S, "model.embed_tokens.weight", &expected, &found, &missing, &mistyped);
    dr_check_opt_quant(&S, "lm_head.weight", &expected, &found, &mistyped);
    dr_check_f32(&S, "model.norm.weight", &expected, &found, &missing, &mistyped);

    for (int i = 0; i < c.layers; i++) {
        snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", i);
        dr_check_f32(&S, nm, &expected, &found, &missing, &mistyped);
        snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", i);
        dr_check_f32(&S, nm, &expected, &found, &missing, &mistyped);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.weight", i);
        dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.weight", i);
        dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.weight", i);
        dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", i);
        dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_norm.weight", i);
        dr_check_f32(&S, nm, &expected, &found, &missing, &mistyped);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_norm.weight", i);
        dr_check_f32(&S, nm, &expected, &found, &missing, &mistyped);
        /* MSA indexer tensors (only expected for layers where sparse_attention_freq[i]==1).
         * index_q_proj/index_k_proj are int4 (need .qs); index_q_norm/index_k_norm are f32. */
        if (c.sparse_freq && c.sparse_freq[i]) {
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.index_q_proj.weight", i);
            dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.index_k_proj.weight", i);
            dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.index_q_norm.weight", i);
            dr_check_f32(&S, nm, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.index_k_norm.weight", i);
            dr_check_f32(&S, nm, &expected, &found, &missing, &mistyped);
        }
        if (i >= c.first_dense) {
            /* MoE layer: router, bias, shared expert, routed experts */
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.weight", i);
            dr_check_f32(&S, nm, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.e_score_correction_bias", i);
            char alt[512];
            snprintf(alt, sizeof(alt), "model.layers.%d.mlp.gate.expert_bias", i);
            dr_check_f32_alt(&S, nm, alt, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_experts.gate_proj.weight", i);
            dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_experts.up_proj.weight", i);
            dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_experts.down_proj.weight", i);
            dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
            for (int e = 0; e < c.experts; e++) {
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.gate_proj.weight", i, e);
                dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.up_proj.weight", i, e);
                dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.down_proj.weight", i, e);
                dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
            }
        } else {
            /* dense MLP layer (first 3) */
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", i);
            dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", i);
            dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", i);
            dr_check_quant(&S, nm, &expected, &found, &missing, &mistyped);
        }
    }

    fprintf(stderr,
            "[dry-run] indexed %d tensors in %d shards, expected %ld, found %ld, "
            "missing %ld, mis-typed %ld\n",
            S.n, S.nfd, expected, found, missing, mistyped);
    if (missing == 0 && mistyped == 0) {
        fprintf(stderr, "[dry-run] all tensors present\n");
        return 0;
    }
    fprintf(stderr, "[dry-run] FAILED: %ld missing, %ld mis-typed\n", missing, mistyped);
    return 1;
}

#ifndef TESTING
int main(int argc, char **argv) {
    /* Set sensible OMP defaults for --bench mode if not already in env. */
    if (!getenv("OMP_PLACES"))    setenv("OMP_PLACES", "cores", 0);
    if (!getenv("OMP_PROC_BIND")) setenv("OMP_PROC_BIND", "close", 0);

    const char *snap = getenv("SNAP");
    if (!snap) {
        fprintf(stderr, "usage: SNAP=<model_dir> ./m3 [cache] [ebits] [dbits] [ctx]\n");
        fprintf(stderr, "       SNAP=<model_dir> ./m3 --dry-run\n");
        fprintf(stderr, "       SNAP=<model_dir> ./m3 --teacher-force [cache] [ebits] [dbits] [ctx]\n");
        fprintf(stderr, "  env: SEED=<n> TF_MODE=1 TF_OUT=<path> TF_TOPK=200\n");
        fprintf(stderr, "       M3_CHECK_NAN=1 DEBUG_TRACE=1 TEMP=<t> TOPP=<p> TOPK=<k>\n");
        return 1;
    }
    /* --dry-run: tensor-presence check before allocating memory */
    if (argc > 1 && !strcmp(argv[1], "--dry-run")) {
        return dry_run(snap);
    }

    /* f8: teacher-force mode flag. Also settable via TF_MODE=1 env var. */
    int argi = 1;
    if (argc > 1 && !strcmp(argv[1], "--teacher-force")) {
        g_tf_mode = 1;
        argi = 2;
    } else if (argc > 1 && !strcmp(argv[1], "--bench")) {
        g_bench_mode = 1;
        argi = 2;
    }

    int cap = argc > argi ? atoi(argv[argi]) : 128;
    int ebits = argc > argi + 1 ? atoi(argv[argi + 1]) : 4;
    int dbits = argc > argi + 2 ? atoi(argv[argi + 2]) : 8;
    int ctx = argc > argi + 3 ? atoi(argv[argi + 3]) : 4096;
    if (getenv("TEMP")) g_temp = (float)atof(getenv("TEMP"));
    if (getenv("TOPP")) g_topp = (float)atof(getenv("TOPP"));
    if (getenv("TOPK")) g_topk_samp = atoi(getenv("TOPK"));
    if (getenv("PLANAR_BITS")) g_planar_bits = atoi(getenv("PLANAR_BITS"));
    if (getenv("TF_MODE")) g_tf_mode = atoi(getenv("TF_MODE"));
    if (getenv("TF_TOPK")) g_tf_topk = atoi(getenv("TF_TOPK"));
    if (getenv("TF_OUT")) g_tf_out_path = getenv("TF_OUT");
    if (getenv("M3_CHECK_NAN")) g_nan_check = atoi(getenv("M3_CHECK_NAN"));
    if (getenv("DEBUG_TRACE")) g_debug_trace = atoi(getenv("DEBUG_TRACE"));
    if (getenv("BENCH_MODE")) g_bench_mode = atoi(getenv("BENCH_MODE"));
    /* f11: VNNI matmul kernels (USE_VNNI=1 enables AVX-512_VNNI path). */
    if (getenv("USE_VNNI")) g_use_vnni = atoi(getenv("USE_VNNI"));
    /* f12: parallel MoE expert dispatch (MOE_PARALLEL=1). */
    if (getenv("MOE_PARALLEL")) g_moe_parallel = atoi(getenv("MOE_PARALLEL"));
    if (getenv("USE_AVX512_I4")) g_use_avx512_i4 = atoi(getenv("USE_AVX512_I4"));
    if (getenv("USE_MTP")) g_use_mtp = atoi(getenv("USE_MTP"));
    g_e3_dir = getenv("E3_DIR");
    /* f10: NUMA topology discovery + optional thread pinning. */
    numa_discover(&g_numa);
    /* Parse env vars AFTER numa_discover (which sets defaults) so they override. */
    if (getenv("NUMA_INTERLEAVE")) {
        g_numa.interleave = atoi(getenv("NUMA_INTERLEAVE"));
    }
    if (getenv("NUMA_PIN_THREADS")) {
        g_numa.pin_threads = atoi(getenv("NUMA_PIN_THREADS"));
    }
    fprintf(stderr, "[f10] NUMA: n_cpus=%d n_nodes=%d interleave=%d pin=%d\n",
            g_numa.n_cpus, g_numa.n_nodes, g_numa.interleave, g_numa.pin_threads);
    if (g_numa.interleave) {
        int br = numa_disable_balancing();
        if (br == 0) fprintf(stderr, "[f10] disabled kernel NUMA balancing\n");
    }
    /* f15: telemetry init (M3_TELEMETRY_PATH env var). */
    telem_init();

    /* f8: seeded determinism. SEED=0 (or unset) => legacy time-seeded path;
     * SEED>0 => fixed seed for reproducible sampling (VAL-CORR-018, VAL-CROSS-006). */
    if (getenv("SEED")) g_seed = atoi(getenv("SEED"));
    if (g_seed > 0) {
        srand((unsigned)g_seed);
        fprintf(stderr, "[seed] %d (fixed)\n", g_seed);
    } else {
        srand((unsigned)time(NULL));
        fprintf(stderr, "[seed] time-seeded\n");
    }

    Model m;
    load_model(&m, snap, ebits, dbits, cap, ctx);
    if (g_use_mtp) e3_load(g_e3_dir);
    fprintf(stderr, "READY\n");

    /* stdin protocol: first line = max_new, second line = space-separated prompt token ids.
     * In teacher-force mode, a third line = space-separated target token ids. */
    char line[1 << 20];
    if (!fgets(line, sizeof(line), stdin)) return 0;
    int max_new = atoi(line);
    if (max_new <= 0) max_new = 128;
    if (!fgets(line, sizeof(line), stdin)) return 0;
    int prompt[8192], np = 0;
    char *p = line, *end = NULL;
    while (np < 8192) {
        long v = strtol(p, &end, 10);
        if (end == p) break;
        prompt[np++] = (int)v;
        p = end;
    }
    if (np < 1) {
        fprintf(stderr, "empty prompt\n");
        return 2;
    }

    if (g_tf_mode) {
        /* teacher-force: read target token IDs (third line) */
        int targets[8192], nt = 0;
        if (!fgets(line, sizeof(line), stdin)) {
            fprintf(stderr, "TF_MODE: missing target token line on stdin\n");
            return 2;
        }
        p = line; end = NULL;
        while (nt < 8192) {
            long v = strtol(p, &end, 10);
            if (end == p) break;
            targets[nt++] = (int)v;
            p = end;
        }
        if (nt < 1) {
            fprintf(stderr, "TF_MODE: empty target list\n");
            return 2;
        }
        if (nt < max_new) max_new = nt;
        fprintf(stderr, "[tf] prompt=%d tokens, targets=%d, topk=%d\n", np, nt, g_tf_topk);
        FILE *tf_out = stdout;
        if (g_tf_out_path && strcmp(g_tf_out_path, "-") != 0) {
            tf_out = fopen(g_tf_out_path, "w");
            if (!tf_out) {
                perror(g_tf_out_path);
                return 2;
            }
        }
        int rc = run_teacher_force(&m, prompt, np, targets, max_new, g_tf_topk, tf_out);
        if (tf_out != stdout) fclose(tf_out);
        return rc;
    }

    if (g_bench_mode) {
        fprintf(stderr, "[bench] prompt=%d tokens, max_new=%d, vnni=%d, moe_parallel=%d\n",
                np, max_new, g_use_vnni, g_moe_parallel);
        return run_bench(&m, prompt, np, max_new);
    }
    fprintf(stderr, "[run] prompt=%d tokens, max_new=%d\n", np, max_new);
    return run_gen(&m, prompt, np, max_new);
}
#endif /* TESTING */
