/* test_rmsnorm.c — VAL-CORR-010: Gemma-style RMSNorm (engine gemma branch).
 *
 * Verifies the engine's rmsnorm() gemma branch (gemma=1) against a NumPy-style
 * reference using the (1+w) * x/rms(x) formula, and asserts the standard
 * (non-Gemma) branch (gemma=0, uses w without +1) produces a *different* result
 * for the same input — proving the gemma branch is taken when use_gemma_norm=1.
 *
 * NOTE: The MiniMax M3 reference (HF `MiniMaxM3VLRMSNorm`, a subclass of
 * `Gemma3RMSNorm`) implements Gemma-style RMSNorm as `output = (1 + weight) *
 * x * rsqrt(mean(x^2) + eps)`. It does NOT subtract the input min — the
 * "subtracts input min" wording in some mission notes is inaccurate; the
 * `use_gemma_norm` flag toggles the `(1+w)` formula (matching the HF reference
 * and the converter's `data_torch + 1.0` convention).
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_rmsnorm.c -o c/tests/test_rmsnorm -lm
 * Run:   ./c/tests/test_rmsnorm
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

/* Reference Gemma-style RMSNorm: o = x * (1/sqrt(mean(x^2)+eps)) * (1+w). */
static void ref_rmsnorm_gemma(double *o, const float *x, const float *w, int D, float eps) {
    double ms = 0.0;
    for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    double r = 1.0 / sqrt(ms / (double)D + (double)eps);
    for (int i = 0; i < D; i++) o[i] = (double)x[i] * r * (1.0 + (double)w[i]);
}

/* Reference standard (non-Gemma) RMSNorm: o = x * (1/sqrt(mean(x^2)+eps)) * w. */
static void ref_rmsnorm_standard(double *o, const float *x, const float *w, int D, float eps) {
    double ms = 0.0;
    for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    double r = 1.0 / sqrt(ms / (double)D + (double)eps);
    for (int i = 0; i < D; i++) o[i] = (double)x[i] * r * (double)w[i];
}

static void test_gemma_branch(void) {
    fprintf(stderr, "==> test_rmsnorm gemma (seq=64, hidden=6144, eps=1e-6)\n");
    const int seq = 64, D = 6144;
    const float eps = 1e-6f;

    float *x = falloc((int64_t)seq * D);
    float *w = falloc(D);
    float *out_g = falloc((int64_t)seq * D);
    float *out_s = falloc((int64_t)seq * D);
    double *ref_g = (double *)malloc((size_t)seq * D * sizeof(double));

    srand(99);
    /* Random weights near 0 (matches HF init: zeros_). */
    for (int i = 0; i < D; i++) w[i] = ((float)(rand() % 2001) - 1000.0f) / 5000.0f;
    /* Random inputs, with one row having an extreme minimum (-1e3) to verify
     * the engine handles large dynamic range correctly under the Gemma formula. */
    for (int64_t i = 0; i < (int64_t)seq * D; i++)
        x[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
    /* Row 5: include a -1e3 outlier to test dynamic range. */
    for (int j = 0; j < D; j++) {
        if (j % 17 == 0)
            x[(int64_t)5 * D + j] = -1.0e3f;
        else
            x[(int64_t)5 * D + j] = (float)(rand() % 100) / 100.0f;
    }

    /* Engine gemma path (gemma=1). */
    for (int s = 0; s < seq; s++)
        rmsnorm(out_g + (int64_t)s * D, x + (int64_t)s * D, w, D, eps, 1);
    /* Engine standard path (gemma=0). */
    for (int s = 0; s < seq; s++)
        rmsnorm(out_s + (int64_t)s * D, x + (int64_t)s * D, w, D, eps, 0);
    /* Reference gemma. */
    for (int s = 0; s < seq; s++)
        ref_rmsnorm_gemma(ref_g + (int64_t)s * D, x + (int64_t)s * D, w, D, eps);

    /* (1) gemma engine path matches gemma reference within 1e-5. */
    float max_diff_g = 0;
    for (int64_t i = 0; i < (int64_t)seq * D; i++) {
        float d = fabsf((float)ref_g[i] - out_g[i]);
        if (d > max_diff_g) max_diff_g = d;
    }
    fprintf(stderr, "   rmsnorm gemma: max abs diff = %.2e (tol 1e-5)\n", max_diff_g);
    CHECK(max_diff_g < 1e-5, "gemma branch matches (1+w) reference within 1e-5");

    /* (2) standard path (gemma=0, uses w not 1+w) is "wrong" for the same input:
     *     the divergence must be substantial (proves the gemma branch is taken
     *     when use_gemma_norm=1, not silently falling through to standard). */
    float max_diff_s_vs_g = 0;
    for (int64_t i = 0; i < (int64_t)seq * D; i++) {
        float d = fabsf(out_s[i] - out_g[i]);
        if (d > max_diff_s_vs_g) max_diff_s_vs_g = d;
    }
    fprintf(stderr, "   standard (gemma=0) vs gemma (gemma=1): max abs diff = %.2e (must be >> 1e-5)\n",
            max_diff_s_vs_g);
    CHECK(max_diff_s_vs_g > 1e-3, "standard path is meaningfully different from gemma path");

    /* (3) Spot-check: row 5 (with the -1e3 outlier) still matches the reference. */
    float max_diff_row5 = 0;
    for (int j = 0; j < D; j++) {
        float d = fabsf((float)ref_g[(int64_t)5 * D + j] - out_g[(int64_t)5 * D + j]);
        if (d > max_diff_row5) max_diff_row5 = d;
    }
    fprintf(stderr, "   row 5 (with -1e3 outlier): max abs diff = %.2e (tol 1e-5)\n", max_diff_row5);
    CHECK(max_diff_row5 < 1e-5, "gemma path handles -1e3 outlier correctly");

    free(x); free(w); free(out_g); free(out_s); free(ref_g);
}

int main(void) {
    fprintf(stderr, "=== rmsnorm unit tests ===\n");
    test_gemma_branch();
    if (g_fail) {
        fprintf(stderr, "\nFAILED: one or more rmsnorm tests failed\n");
        return 1;
    }
    fprintf(stderr, "\nPASS: all rmsnorm tests passed\n");
    return 0;
}
