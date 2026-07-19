/* test_swiglu.c — VAL-CORR-011: SwiGLUOAI activation (alpha=1.702, clamp=7.0).
 *
 * Verifies the engine's swiglu() scalar kernel matches a NumPy-style reference
 * using the MiniMax M3 SwiGLUOAI formulation: clamp(x, [-lim, lim]) *
 * sigmoid(alpha * clamp(x, [-lim, lim])). Asserts:
 *   (a) the clamp fires for |x| > 7.0 (output saturates),
 *   (b) alpha scaling is applied near zero (output differs from plain silu),
 *   (c) plain swiglu (no alpha, no clamp) produces a *different* answer at
 *       the clamp boundary, proving the SwiGLUOAI branch is exercised.
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

/* Reference SwiGLUOAI gate (matching engine's swiglu() kernel):
 *   g = clamp(x, [-lim, lim])
 *   out = g * sigmoid(alpha * g)
 * Uses double precision to maximize reference accuracy. */
static double ref_swiglu(double x, double alpha, double lim) {
    double g = x;
    if (g > lim) g = lim;
    if (g < -lim) g = -lim;
    double s = 1.0 / (1.0 + exp(-alpha * g));
    return g * s;
}

/* Plain SwiGLU (no alpha, no clamp): out = silu(x) = x * sigmoid(x).
 * This is what a non-SwiGLUOAI implementation would compute — it must
 * produce a different answer at the clamp boundary. */
static double ref_plain_silu(double x) {
    return x / (1.0 + exp(-x));
}

static void test_swigluoai(void) {
    fprintf(stderr, "==> test_swiglu (alpha=1.702, clamp=7.0)\n");
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
         * (to exercise the clamp). */
        float v;
        int bucket = rand() % 3;
        if (bucket == 0)      v = ((float)(rand() % 2001) - 1000.0f) / 100.0f;   /* ±10 */
        else if (bucket == 1) v = ((float)(rand() % 2001) - 1000.0f) / 1.0e5f;   /* ~0 (alpha test) */
        else                  v = ((float)(rand() % 3001) - 1500.0f) / 10.0f;     /* ±150 (clamp test) */
        gate[i] = v;
        up[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
    }

    /* Engine: out = swiglu(gate, alpha, lim) * up. */
    for (int64_t i = 0; i < N; i++)
        out_eng[i] = swiglu(gate[i], (float)alpha, (float)lim) * up[i];
    /* Reference (double). */
    for (int64_t i = 0; i < N; i++)
        out_ref[i] = ref_swiglu((double)gate[i], alpha, lim) * (double)up[i];
    /* Plain swiglu (no alpha, no clamp). */
    for (int64_t i = 0; i < N; i++)
        out_plain[i] = ref_plain_silu((double)gate[i]) * (double)up[i];

    /* (1) engine matches reference within 1e-5. */
    float max_diff = 0;
    for (int64_t i = 0; i < N; i++) {
        float d = fabsf((float)out_ref[i] - out_eng[i]);
        if (d > max_diff) max_diff = d;
    }
    fprintf(stderr, "   swigluoai: max abs diff = %.2e (tol 1e-5), alpha=1.702, clamp=7.0\n", max_diff);
    CHECK(max_diff < 1e-5, "SwiGLUOAI matches reference within 1e-5");

    /* (2) Verify clamp fires: pick a gate value > 7 and check the output matches
     *     the clamped reference exactly (within fp tolerance). */
    float big_gate = 50.0f;
    float eng_big = swiglu(big_gate, (float)alpha, (float)lim);
    float ref_big_clamped = (float)ref_swiglu(7.0, alpha, lim);   /* clamp to 7.0 */
    float ref_big_unclamped = (float)ref_swiglu(50.0, alpha, lim); /* should also == clamped */
    CHECK(fabsf(eng_big - ref_big_clamped) < 1e-6, "clamp fires for x=50 (output == x=7)");
    CHECK(fabsf(ref_big_unclamped - ref_big_clamped) < 1e-6, "reference clamps x=50 to x=7");
    fprintf(stderr, "   clamp boundary: swiglu(50) = %.6f, swiglu(7) = %.6f (match)\n",
            eng_big, ref_big_clamped);

    /* (3) alpha scaling fires near zero: swiglu(0.5, alpha=1.702) != silu(0.5). */
    float small = 0.5f;
    float eng_small = swiglu(small, (float)alpha, (float)lim);
    float plain_small = (float)ref_plain_silu(small);
    CHECK(fabsf(eng_small - plain_small) > 1e-3, "alpha scaling differs from plain silu near zero");
    fprintf(stderr, "   alpha scaling: swiglu(0.5) = %.6f, silu(0.5) = %.6f (differ)\n",
            eng_small, plain_small);

    /* (4) Plain swiglu (no alpha, no clamp) is "wrong" at clamp boundary:
     *     the divergence must be substantial. */
    float max_diff_plain = 0;
    for (int64_t i = 0; i < N; i++) {
        float d = fabsf((float)out_plain[i] - (float)out_ref[i]);
        if (d > max_diff_plain) max_diff_plain = d;
    }
    fprintf(stderr, "   plain swiglu vs SwiGLUOAI: max abs diff = %.2e (must be >> 1e-5 at clamp boundary)\n",
            max_diff_plain);
    CHECK(max_diff_plain > 1e-3, "plain swiglu (no alpha/clamp) is meaningfully different");

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
