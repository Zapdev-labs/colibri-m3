/* test_moe_routing.c — VAL-CORR-012: sigmoid + e_score_correction_bias + route_norm + top-4 of 128.
 *
 * Verifies the engine's moe_router() function (extracted from moe()) against a
 * NumPy-style reference implementing the HF MiniMaxM3VLTopKRouter algorithm:
 *   logit   = x @ router.T                 (f32 router, [E, D])
 *   score   = sigmoid(logit)               (sigmoid scoring, NOT softmax)
 *   choice  = score + e_score_correction_bias   (used for top-k selection)
 *   topk_idx = top-K(choice)               (top-4 of 128)
 *   topk_w  = score[topk_idx]              (router weight = sigmoid score, NOT choice)
 *   topk_w /= sum(topk_w)                  (route_norm normalization)
 *   topk_w *= router_scale                  (router_scale = 2.0)
 * Plus the always-on shared expert (always dispatched by moe()).
 *
 * Asserts:
 *   (a) exactly 4 experts are selected,
 *   (b) the bias shifts selections (with-bias vs zero-bias picks differ),
 *   (c) the router weights sum to router_scale=2.0 over the 4 selected (post-norm),
 *   (d) the per-expert sigmoid score (NOT score+bias) is used as the weight,
 *   (e) the shared expert is always dispatched (engine moe() always computes
 *       shared output — verified by a smoke check that moe() runs the shared
 *       expert path even when no routed experts are cached).
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_moe_routing.c -o c/tests/test_moe_routing -lm
 * Run:   ./c/tests/test_moe_routing
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

/* Reference MoE router implementing HF MiniMaxM3VLTopKRouter. */
static void ref_moe_router(int *idx, double *w, const float *x, const float *router,
                           const float *bias, int D, int E, int K,
                           int route_norm, double router_scale) {
    double *logit = (double *)malloc((size_t)E * sizeof(double));
    double *score = (double *)malloc((size_t)E * sizeof(double));
    double *choice = (double *)malloc((size_t)E * sizeof(double));
    for (int e = 0; e < E; e++) {
        double s = 0.0;
        for (int i = 0; i < D; i++) s += (double)x[i] * (double)router[(int64_t)e * D + i];
        logit[e] = s;
        score[e] = 1.0 / (1.0 + exp(-s));                /* sigmoid */
        choice[e] = score[e] + (double)bias[e];         /* + bias for selection */
    }
    for (int kk = 0; kk < K; kk++) {
        int best = -1;
        double bv = -1e300;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int j = 0; j < kk; j++) if (idx[j] == e) { taken = 1; break; }
            if (!taken && choice[e] > bv) { bv = choice[e]; best = e; }
        }
        idx[kk] = best;
        w[kk] = score[best];                              /* weight = sigmoid score, NOT choice */
    }
    if (route_norm) {
        double sm = 1e-20;
        for (int kk = 0; kk < K; kk++) sm += w[kk];
        for (int kk = 0; kk < K; kk++) w[kk] /= sm;
    }
    for (int kk = 0; kk < K; kk++) w[kk] *= router_scale;
    free(logit); free(score); free(choice);
}

/* Build a minimal Model with a sparse (MoE) layer so we can drive moe() end-to-end
 * and verify the shared expert path runs unconditionally. */
static void build_minimal_moe_model(Model *m, Cfg *c, int D, int E, int K, int I, int sI) {
    memset(m, 0, sizeof(*m));
    memset(c, 0, sizeof(*c));
    c->hidden = D; c->experts = E; c->topk = K; c->moe_inter = I;
    c->n_shared = 1; c->router_scale = 2.0f; c->route_norm = 1;
    c->sw_alpha = 1.702f; c->sw_limit = 7.0f; c->eps = 1e-6f; c->layers = 1;
    m->c = *c;
    m->ebits = 4; m->dbits = 4; m->ecap = 4; m->max_t = 16;
    m->L = (Layer *)calloc(1, sizeof(Layer));
    Layer *l = &m->L[0];
    l->sparse = 1;
    /* Router + bias (f32). */
    l->router = falloc((int64_t)E * D);
    l->router_bias = falloc(E);
    srand(2025);
    for (int i = 0; i < E * D; i++) l->router[i] = ((float)(rand() % 2001) - 1000.0f) / 500.0f;
    for (int e = 0; e < E; e++) l->router_bias[e] = ((float)(rand() % 2001) - 1000.0f) / 200.0f;
    /* Shared expert: gate/up [sI, D], down [D, sI] (f32 for test simplicity). */
    l->sh_gate.qf = falloc((int64_t)sI * D); l->sh_gate.fmt = 0; l->sh_gate.O = sI; l->sh_gate.I = D;
    l->sh_up.qf   = falloc((int64_t)sI * D); l->sh_up.fmt   = 0; l->sh_up.O   = sI; l->sh_up.I   = D;
    l->sh_down.qf = falloc((int64_t)D * sI); l->sh_down.fmt = 0; l->sh_down.O = D; l->sh_down.I = sI;
    for (int i = 0; i < sI * D; i++) l->sh_gate.qf[i] = 0.0f;
    for (int i = 0; i < sI * D; i++) l->sh_up.qf[i]   = 0.0f;
    for (int i = 0; i < D * sI; i++) l->sh_down.qf[i]  = 0.0f;
    /* Expert cache (zeroed — expert_get will load on demand; for the shared-only
     * smoke check we don't need real expert weights, just the shared path firing). */
    m->cache = (ESlot **)calloc(1, sizeof(ESlot *));
    m->cn = (int *)calloc(1, sizeof(int));
    m->cache[0] = (ESlot *)calloc(m->ecap, sizeof(ESlot));
}

static void test_router_basic(void) {
    fprintf(stderr, "==> test_moe_router basic (D=6144, E=128, K=4)\n");
    const int D = 6144, E = 128, K = 4;
    const int route_norm = 1;
    const float router_scale = 2.0f;

    float *x = falloc(D);
    float *router = falloc((int64_t)E * D);
    float *bias = falloc(E);
    float *logit = falloc(E), *choice = falloc(E);
    srand(314);
    for (int i = 0; i < D; i++) x[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
    for (int i = 0; i < E * D; i++) router[i] = ((float)(rand() % 2001) - 1000.0f) / 500.0f;
    for (int e = 0; e < E; e++) bias[e] = ((float)(rand() % 2001) - 1000.0f) / 200.0f;

    int idx_eng[4]; float w_eng[4];
    int idx_ref[4]; double w_ref[4];
    moe_router(idx_eng, w_eng, x, router, bias, D, E, K, route_norm, router_scale, logit, choice);
    ref_moe_router(idx_ref, w_ref, x, router, bias, D, E, K, route_norm, (double)router_scale);

    /* (a) exactly 4 experts selected (all idx >= 0 and distinct). */
    int distinct = 1;
    for (int i = 0; i < K; i++) {
        if (idx_eng[i] < 0) distinct = 0;
        for (int j = i + 1; j < K; j++) if (idx_eng[i] == idx_eng[j]) distinct = 0;
    }
    CHECK(distinct, "exactly 4 distinct experts selected");
    fprintf(stderr, "   top4 selected (engine):");
    for (int kk = 0; kk < K; kk++) fprintf(stderr, " %d", idx_eng[kk]);
    fprintf(stderr, "\n");

    /* Engine picks must match reference picks (as a set; order may differ for ties). */
    int set_match = 1;
    for (int i = 0; i < K; i++) {
        int found = 0;
        for (int j = 0; j < K; j++) if (idx_eng[j] == idx_ref[i]) { found = 1; break; }
        if (!found) set_match = 0;
    }
    CHECK(set_match, "engine picks match reference top-4 set");

    /* (c) router weights sum to router_scale=2.0 over the 4 selected. */
    float sum_w = 0;
    for (int kk = 0; kk < K; kk++) sum_w += w_eng[kk];
    fprintf(stderr, "   router weights (engine):");
    for (int kk = 0; kk < K; kk++) fprintf(stderr, " %.4f", w_eng[kk]);
    fprintf(stderr, " sum=%.4f (scale=%.1f)\n", sum_w, router_scale);
    CHECK(fabsf(sum_w - router_scale) < 1e-4, "router weights sum to router_scale=2.0");

    /* (d) per-expert weight matches the sigmoid score (NOT score+bias): verify
     *     by recomputing sigmoid(logit) for each selected expert and checking
     *     w[kk] = sigmoid(raw_logit[best]) / sum_sig * router_scale, where
     *     sum_sig is the sum of the 4 selected sigmoid scores. */
    double sig[4];
    double sum_sig = 0;
    for (int kk = 0; kk < K; kk++) {
        int e = idx_eng[kk];
        double s = 0.0;
        for (int i = 0; i < D; i++) s += (double)x[i] * (double)router[(int64_t)e * D + i];
        sig[kk] = 1.0 / (1.0 + exp(-s));
        sum_sig += sig[kk];
    }
    int w_is_sigmoid = 1;
    for (int kk = 0; kk < K; kk++) {
        /* Recover the raw sigmoid score: w[kk] = sig[kk] / sum_sig * router_scale. */
        double recovered = (double)w_eng[kk] * sum_sig / (double)router_scale;
        if (fabs(recovered - sig[kk]) > 1e-3) w_is_sigmoid = 0;
    }
    CHECK(w_is_sigmoid, "router weight is sigmoid score (not sigmoid+bias)");

    free(x); free(router); free(bias); free(logit); free(choice);
}

static void test_bias_shifts_selections(void) {
    fprintf(stderr, "==> test_moe_router bias shifts selections\n");
    const int D = 6144, E = 128, K = 4;
    const int route_norm = 1;
    const float router_scale = 2.0f;

    float *x = falloc(D);
    float *router = falloc((int64_t)E * D);
    float *bias_real = falloc(E);
    float *bias_zero = falloc(E);
    float *logit = falloc(E), *choice = falloc(E);
    srand(777);
    for (int i = 0; i < D; i++) x[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
    for (int i = 0; i < E * D; i++) router[i] = ((float)(rand() % 2001) - 1000.0f) / 500.0f;
    for (int e = 0; e < E; e++) {
        bias_real[e] = ((float)(rand() % 2001) - 1000.0f) / 100.0f;  /* ±10 — large enough to shift */
        bias_zero[e] = 0.0f;
    }

    int idx_with_bias[4], idx_no_bias[4];
    float w_with_bias[4], w_no_bias[4];
    moe_router(idx_with_bias, w_with_bias, x, router, bias_real, D, E, K, route_norm, router_scale, logit, choice);
    moe_router(idx_no_bias,  w_no_bias,  x, router, bias_zero, D, E, K, route_norm, router_scale, logit, choice);

    int differ = 0;
    for (int i = 0; i < K; i++) {
        int found = 0;
        for (int j = 0; j < K; j++) if (idx_with_bias[j] == idx_no_bias[i]) { found = 1; break; }
        if (!found) differ = 1;
    }
    fprintf(stderr, "   with-bias picks:");
    for (int i = 0; i < K; i++) fprintf(stderr, " %d", idx_with_bias[i]);
    fprintf(stderr, "\n   zero-bias picks:");
    for (int i = 0; i < K; i++) fprintf(stderr, " %d", idx_no_bias[i]);
    fprintf(stderr, "\n   selections differ: %s\n", differ ? "YES" : "NO");
    CHECK(differ, "e_score_correction_bias shifts the top-4 selection");

    free(x); free(router); free(bias_real); free(bias_zero); free(logit); free(choice);
}

static void test_shared_expert_always_dispatched(void) {
    fprintf(stderr, "==> test_moe shared expert always dispatched\n");
    /* Drive moe() end-to-end on a minimal Model. The shared expert's gate/up
     * are zeroed, so the shared output contributes nothing — but the engine
     * must still execute the shared path (compute sg, su, swiglu, hh, accumulate).
     * We verify this by setting the shared gate/up to NON-zero constants and
     * checking the output is non-zero (proving the shared path ran). */
    const int D = 256, E = 8, K = 4, I = 64, sI = I;  /* small for speed */
    Model m; Cfg c;
    build_minimal_moe_model(&m, &c, D, E, K, I, sI);
    Layer *l = &m.L[0];
    /* Override shared expert weights to identity-like so the output is
     * predictable: sh_gate[r, r] = 1, sh_up[r, r] = 1, sh_down[r, r] = 1
     * (with sh_gate/sh_up as [sI, D], sh_down as [D, sI]).
     * Then matmul(sh_gate, x)[r] = x[r], matmul(sh_up, x)[r] = x[r],
     * swiglu(gate=x[r], up=x[r]) = (up+1)*glu (HF _apply_gate), and
     * matmul(sh_down, .)[r] = swiglu(x[r], x[r]). */
    for (int i = 0; i < sI * D; i++) { l->sh_gate.qf[i] = 0.0f; l->sh_up.qf[i] = 0.0f; }
    for (int r = 0; r < sI && r < D; r++) {
        l->sh_gate.qf[(int64_t)r * D + r] = 1.0f;
        l->sh_up.qf[(int64_t)r * D + r] = 1.0f;
    }
    for (int i = 0; i < D * sI; i++) l->sh_down.qf[i] = 0.0f;
    for (int r = 0; r < sI && r < D; r++) l->sh_down.qf[(int64_t)r * sI + r] = 1.0f;
    /* Set router so the top-4 picks land on experts 0..3 — but we won't load
     * real expert weights, so we stub expert_get to return zeroed ESlots by
     * pre-populating the cache with zeroed slots for experts 0..3. */
    for (int e = 0; e < 4; e++) {
        ESlot *s = &m.cache[0][e];
        s->eid = e;
        s->g.qf = falloc((int64_t)I * D); s->g.fmt = 0; s->g.O = I; s->g.I = D;
        s->u.qf = falloc((int64_t)I * D); s->u.fmt = 0; s->u.O = I; s->u.I = D;
        s->d.qf = falloc((int64_t)D * I); s->d.fmt = 0; s->d.O = D; s->d.I = I;
        for (int i = 0; i < I * D; i++) { s->g.qf[i] = 0.0f; s->u.qf[i] = 0.0f; }
        for (int i = 0; i < D * I; i++) s->d.qf[i] = 0.0f;
        s->used = ++m.clock;
    }
    m.cn[0] = 4;
    /* Bias router to pick experts 0..3 by giving them very high bias. */
    for (int e = 0; e < E; e++) l->router_bias[e] = (e < 4) ? 100.0f : -100.0f;

    float *x = falloc(D);
    float *out = falloc(D);
    for (int i = 0; i < D; i++) x[i] = 1.0f;
    for (int i = 0; i < D; i++) out[i] = 0.0f;
    moe(&m, l, 0, x, 1, out);

    /* Shared expert path with identity weights:
     *   sg = sh_gate @ x  -> sg[r] = x[r] = 1.0 (for r < sI)
     *   su = sh_up   @ x  -> su[r] = x[r] = 1.0
     *   swiglu(gate=1.0, up=1.0, alpha=1.702, lim=7.0) = (1.0+1.0)*sigmoid(1.702)
     *                                                   ≈ 1.6916  (HF _apply_gate)
     *   hh = sh_down @ sg' -> hh[r] = swiglu(1.0, 1.0) (for r < sI)
     * So out[r] = swiglu(1.0, 1.0) for r < sI. */
    float expected = swiglu(1.0f, 1.0f, c.sw_alpha, c.sw_limit);
    int shared_ran = 0;
    for (int r = 0; r < sI && r < D; r++) {
        if (fabsf(out[r] - expected) < 1e-4) shared_ran = 1;
    }
    fprintf(stderr, "   moe() output[0] = %.6f, expected shared contribution = %.6f (match => shared dispatched)\n",
            out[0], expected);
    CHECK(shared_ran, "shared expert path runs unconditionally (always dispatched)");

    /* Cleanup. */
    for (int e = 0; e < 4; e++) {
        free(m.cache[0][e].g.qf); free(m.cache[0][e].u.qf); free(m.cache[0][e].d.qf);
    }
    free(m.cache[0]); free(m.cache); free(m.cn);
    free(l->router); free(l->router_bias);
    free(l->sh_gate.qf); free(l->sh_up.qf); free(l->sh_down.qf);
    free(m.L); free(x); free(out);
}

int main(void) {
    fprintf(stderr, "=== moe routing unit tests ===\n");
    test_router_basic();
    test_bias_shifts_selections();
    test_shared_expert_always_dispatched();
    if (g_fail) {
        fprintf(stderr, "\nFAILED: one or more moe routing tests failed\n");
        return 1;
    }
    fprintf(stderr, "\nPASS: all moe routing tests passed\n");
    return 0;
}
