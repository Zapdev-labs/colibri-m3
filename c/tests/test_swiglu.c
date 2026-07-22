/* test_swiglu.c — VAL-CORR-011: SwiGLUOAI activation (alpha=1.702, clamp=7.0).
 *
 * Verifies the engine's swiglu() scalar kernel matches a NumPy-style reference
 * implementing the HF MiniMaxM3VLDenseMLP._apply_gate / MiniMaxM3VLExperts._apply_gate
 * formula EXACTLY (see transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py):
 *
 *     gate = gate.clamp(max=lim)             # asymmetric: upper-only clamp
 *     up   = up.clamp(min=-lim, max=lim)     # symmetric clamp on `up`
 *     glu  = gate * sigmoid(gate * alpha)
 *     out  = (up + 1.0) * glu
 *
 * The engine's swiglu(gate, up, alpha, lim) returns the full (up+1)*glu value.
 * The test asserts:
 *   (a) the engine matches a double-precision reference within 1e-5,
 *   (b) the asymmetric gate clamp fires for gate > +lim (but NOT for gate < -lim),
 *   (c) the symmetric up clamp fires for |up| > lim,
 *   (d) the (up + 1.0) multiplier is applied (out != glu * up),
 *   (e) alpha scaling is applied near zero (out differs from plain silu(gate)*up),
 *   (f) plain SwiGLU (silu(gate) * up, no alpha, no clamp, no +1.0) produces a
 *       *wrong* answer, proving the SwiGLUOAI branch is exercised.
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_swiglu.c -o c/tests/test_swiglu -lm
 * Run:   ./c/tests/test_swiglu
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

/* Reference SwiGLUOAI _apply_gate (matching HF MiniMaxM3VLDenseMLP._apply_gate):
 *   g = clamp(gate, max=lim)              // asymmetric: upper-only
 *   u = clamp(up, min=-lim, max=lim)      // symmetric on up
 *   glu = g * sigmoid(g * alpha)
 *   out = (u + 1.0) * glu
 * Uses double precision to maximize reference accuracy. */
static double ref_swiglu(double gate, double up, double alpha, double lim) {
    double g = gate > lim ? lim : gate;                       /* clamp(max=lim) */
    double u = up > lim ? lim : (up < -lim ? -lim : up);      /* clamp([-lim,lim]) */
    double glu = g * (1.0 / (1.0 + exp(-alpha * g)));
    return (u + 1.0) * glu;
}

/* Plain SwiGLU (no alpha, no clamp, no +1.0): out = silu(gate) * up =
 * gate * sigmoid(gate) * up. This is what a non-SwiGLUOAI implementation
 * would compute — it must produce a *different* answer. */
static double ref_plain_silu_mul_up(double gate, double up) {
    return (gate / (1.0 + exp(-gate))) * up;
}

/* glu only (no +1.0, no up clamp): out = glu * up. Used to prove the (up+1.0)
 * multiplier branch is taken. */
static double ref_glu_mul_up(double gate, double up, double alpha, double lim) {
    double g = gate > lim ? lim : gate;
    double glu = g * (1.0 / (1.0 + exp(-alpha * g)));
    return glu * up;
}

static void test_swigluoai(void) {
    fprintf(stderr, "==> test_swiglu (alpha=1.702, clamp=7.0, HF _apply_gate)\n");
    const double alpha = 1.702;
    const double lim = 7.0;
    const int seq = 64, inter = 3072;
    const int64_t N = (int64_t)seq * inter;

    float *gate = falloc(N);
    float *up = falloc(N);
    float *out_eng = falloc(N);
    double *out_ref = (double *)malloc((size_t)N * sizeof(double));
    double *out_plain = (double *)malloc((size_t)N * sizeof(double));

    srand(5);
    for (int64_t i = 0; i < N; i++) {
        /* Mix of normal-range values, near-zero values, and large outliers
         * (to exercise both gate and up clamps). */
        float v;
        int bucket = rand() % 3;
        if (bucket == 0)      v = ((float)(rand() % 2001) - 1000.0f) / 100.0f;   /* ±10 */
        else if (bucket == 1) v = ((float)(rand() % 2001) - 1000.0f) / 1.0e5f;   /* ~0 (alpha test) */
        else                  v = ((float)(rand() % 3001) - 1500.0f) / 10.0f;     /* ±150 (clamp test) */
        gate[i] = v;
        up[i] = ((float)(rand() % 3001) - 1500.0f) / 10.0f;   /* ±150 (up clamp test) */
    }

    /* Engine: out = swiglu(gate, up, alpha, lim) = (up_clamped + 1.0) * glu. */
    for (int64_t i = 0; i < N; i++)
        out_eng[i] = swiglu(gate[i], up[i], (float)alpha, (float)lim);
    /* Reference (double). */
    for (int64_t i = 0; i < N; i++)
        out_ref[i] = ref_swiglu((double)gate[i], (double)up[i], alpha, lim);
    /* Plain swiglu (no alpha, no clamp, no +1.0). */
    for (int64_t i = 0; i < N; i++)
        out_plain[i] = ref_plain_silu_mul_up((double)gate[i], (double)up[i]);

    /* (1) engine matches reference within 1e-5. */
    float max_diff = 0;
    for (int64_t i = 0; i < N; i++) {
        float d = fabsf((float)out_ref[i] - out_eng[i]);
        if (d > max_diff) max_diff = d;
    }
    fprintf(stderr, "   swigluoai: max abs diff = %.2e (tol 1e-5), alpha=1.702, clamp=7.0\n", max_diff);
    CHECK(max_diff < 1e-5, "SwiGLUOAI matches HF _apply_gate reference within 1e-5");

    /* (2) Asymmetric gate clamp: HF uses clamp(max=lim) — upper-only, NO lower
     *     clamp. For gate > +lim the upper clamp fires (gate=50 == gate=7).
     *     For gate < -lim the engine must match the asymmetric reference
     *     (gate NOT lower-clamped), and must NOT match a symmetric-clamp
     *     reference (which would clamp gate=-50 to gate=-7). Because
     *     sigmoid(alpha*gate) saturates for large |gate|, the numerical
     *     difference between asymmetric and symmetric at gate=-50 is small
     *     (~7e-5), so we assert the engine matches the asymmetric reference
     *     exactly and differs from the symmetric-clamp reference. */
    float upv = 0.5f;  /* small up so the up clamp does not fire */
    float eng_big = swiglu(50.0f, upv, (float)alpha, (float)lim);
    float ref_big_7 = (float)ref_swiglu(7.0, upv, alpha, lim);   /* upper clamp fires */
    CHECK(fabsf(eng_big - ref_big_7) < 1e-5, "gate clamp(max=lim) fires for gate=50 (== gate=7)");
    fprintf(stderr, "   gate clamp(max=lim): swiglu(50,%.2f) = %.6f, swiglu(7,%.2f) = %.6f (match)\n",
            upv, eng_big, upv, ref_big_7);

    /* Asymmetric: gate=-50 must match the asymmetric (no-lower-clamp) reference
     * and must NOT match the symmetric-clamp (gate=-7) reference. */
    float eng_neg = swiglu(-50.0f, upv, (float)alpha, (float)lim);
    float ref_asym_neg = (float)ref_swiglu(-50.0, upv, alpha, lim);  /* HF: no lower clamp */
    float ref_sym_neg  = (float)ref_swiglu(-7.0, upv, alpha, lim);   /* symmetric would clamp to -7 */
    CHECK(fabsf(eng_neg - ref_asym_neg) < 1e-6, "engine matches asymmetric (no lower clamp) for gate=-50");
    CHECK(fabsf(eng_neg - ref_sym_neg) > 1e-6, "engine does NOT symmetric-clamp gate=-50 to gate=-7");
    CHECK(fabsf(ref_asym_neg - ref_sym_neg) > 1e-6, "asymmetric vs symmetric gate clamp differ at gate=-50");
    fprintf(stderr, "   gate clamp asymmetric: swiglu(-50,%.2f) = %.6e (asym), swiglu(-7,%.2f) = %.6e (sym, differs)\n",
            upv, eng_neg, upv, ref_sym_neg);

    /* (3) Symmetric up clamp fires for |up| > lim: swiglu(gate, 50, ...) ==
     *     swiglu(gate, 7, ...) and swiglu(gate, -50, ...) == swiglu(gate, -7, ...). */
    float gv = 0.5f;  /* small gate so the gate clamp does not fire */
    float eng_up_big = swiglu(gv, 50.0f, (float)alpha, (float)lim);
    float ref_up_7   = (float)ref_swiglu(gv, 7.0, alpha, lim);
    CHECK(fabsf(eng_up_big - ref_up_7) < 1e-5, "up clamp(max=lim) fires for up=50 (== up=7)");
    float eng_up_negbig = swiglu(gv, -50.0f, (float)alpha, (float)lim);
    float ref_up_neg7   = (float)ref_swiglu(gv, -7.0, alpha, lim);
    CHECK(fabsf(eng_up_negbig - ref_up_neg7) < 1e-5, "up clamp(min=-lim) fires for up=-50 (== up=-7)");
    fprintf(stderr, "   up clamp: swiglu(%.2f,50) = %.6f == swiglu(%.2f,7) = %.6f; "
            "swiglu(%.2f,-50) = %.6f == swiglu(%.2f,-7) = %.6f\n",
            gv, eng_up_big, gv, ref_up_7, gv, eng_up_negbig, gv, ref_up_neg7);

    /* (4) (up + 1.0) multiplier is applied: engine output != glu * up (without +1).
     *     Pick a gate/up that do NOT clamp so the only difference is the +1.0. */
    float gv2 = 0.3f, uv2 = 0.4f;
    float eng_applied = swiglu(gv2, uv2, (float)alpha, (float)lim);
    float glu_mul_up  = (float)ref_glu_mul_up(gv2, uv2, alpha, lim);
    CHECK(fabsf(eng_applied - glu_mul_up) > 1e-3, "(up+1.0) multiplier applied (out != glu*up)");
    /* And it should equal glu * (up + 1.0). */
    float glu_plus_one = (float)ref_swiglu(gv2, uv2, alpha, lim);
    CHECK(fabsf(eng_applied - glu_plus_one) < 1e-5, "out == (up+1.0)*glu exactly");
    fprintf(stderr, "   +1.0 multiplier: swiglu(%.2f,%.2f) = %.6f, glu*up = %.6f, (up+1)*glu = %.6f\n",
            gv2, uv2, eng_applied, glu_mul_up, glu_plus_one);

    /* (5) alpha scaling fires near zero: swiglu(0.5, up, alpha, ...) != silu(0.5)*up. */
    float small_g = 0.5f, small_u = 0.5f;
    float eng_small = swiglu(small_g, small_u, (float)alpha, (float)lim);
    float plain_small = (float)ref_plain_silu_mul_up(small_g, small_u);
    CHECK(fabsf(eng_small - plain_small) > 1e-3, "alpha scaling differs from plain silu*up near zero");
    fprintf(stderr, "   alpha scaling: swiglu(0.5,0.5) = %.6f, silu(0.5)*0.5 = %.6f (differ)\n",
            eng_small, plain_small);

    /* (6) Plain swiglu (silu(gate)*up, no alpha/clamp/+1.0) is "wrong" over the
     *     full fixture: the divergence must be substantial. */
    float max_diff_plain = 0;
    for (int64_t i = 0; i < N; i++) {
        float d = fabsf((float)out_plain[i] - (float)out_ref[i]);
        if (d > max_diff_plain) max_diff_plain = d;
    }
    fprintf(stderr, "   plain swiglu vs SwiGLUOAI: max abs diff = %.2e (must be >> 1e-5 at clamp boundary)\n",
            max_diff_plain);
    CHECK(max_diff_plain > 1e-3, "plain swiglu (no alpha/clamp/+1.0) is meaningfully different");

    free(gate); free(up); free(out_eng); free(out_ref); free(out_plain);
}

int main(void) {
    fprintf(stderr, "=== swiglu unit tests ===\n");
    test_swigluoai();
    if (g_fail) {
        fprintf(stderr, "\nFAILED: one or more swiglu tests failed\n");
        return 1;
    }
    fprintf(stderr, "\nPASS: all swiglu tests passed\n");
    return 0;
}
