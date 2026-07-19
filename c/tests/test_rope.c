/* test_rope.c — VAL-CORR-009: partial RoPE with theta=5e6, rotary_dim=64.
 *
 * Verifies the engine's rope() kernel against a NumPy-style rotate-half
 * reference on a [seq=128, heads=64, head_dim=128] Q tensor with theta=5e6,
 * rotary_dim=64 (partial_rotary_factor=0.5). Asserts:
 *   (a) dims 64-127 (the "nope" half) are unmodified,
 *   (b) the rotation direction matches HF's rotate_half convention
 *       (x1=c*x1 - s*x2, x2=c*x2 + s*x1) — not the inverted-sign convention.
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_rope.c -o c/tests/test_rope -lm
 * Run:   ./c/tests/test_rope
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

/* Reference partial RoPE applying rotate-half to the first rotary_dim channels
 * using HF's rotate_half convention (matches apply_rotary_pos_emb in HF transformers).
 * HF: rotate_half(x) = cat(-x2, x1) where x1=x[:h2], x2=x[h2:rd].
 * Final: x1_out = x1*cos - x2*sin; x2_out = x2*cos + x1*sin. */
static void ref_rope(float *x, int pos, double theta, int hd, int rd) {
    if (rd <= 0 || rd > hd) rd = hd;
    int h2 = rd / 2;
    for (int j = 0; j < h2; j++) {
        double inv = pow(theta, -2.0 * (double)j / (double)rd);
        double ang = (double)pos * inv;
        double c = cos(ang), s = sin(ang);
        double x1 = x[j], x2 = x[j + h2];
        x[j]       = (float)(x1 * c - x2 * s);
        x[j + h2]  = (float)(x2 * c + x1 * s);
    }
    /* dims rd..hd-1 are unmodified (nope half) */
}

static void test_partial_rope(void) {
    fprintf(stderr, "==> test_rope partial (theta=5e6, rotary_dim=64, head_dim=128)\n");
    const int seq = 128, heads = 64, hd = 128;
    const int rotary_dim = 64;
    const float theta = 5.0e6f;

    int64_t n = (int64_t)seq * heads * hd;
    float *Q = falloc(n), *Qref = falloc(n);
    srand(42);
    for (int64_t i = 0; i < n; i++) {
        float v = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;  /* ±1 */
        Q[i] = v;
        Qref[i] = v;
    }

    /* Apply engine rope to each (seq, head) vector. */
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            int pos = s;  /* position id == sequence index */
            float *xh = Q + ((int64_t)s * heads + h) * hd;
            rope(xh, pos, theta, hd, rotary_dim);
        }
    }
    /* Apply reference rope. */
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            float *xh = Qref + ((int64_t)s * heads + h) * hd;
            ref_rope(xh, s, (double)theta, hd, rotary_dim);
        }
    }

    /* Max abs diff over the rotated half (dims 0..63). */
    float max_diff_rot = 0;
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            const float *a = Q + ((int64_t)s * heads + h) * hd;
            const float *b = Qref + ((int64_t)s * heads + h) * hd;
            for (int j = 0; j < rotary_dim; j++) {
                float d = fabsf(a[j] - b[j]);
                if (d > max_diff_rot) max_diff_rot = d;
            }
        }
    }

    /* The nope half (dims 64..127) must be unmodified. */
    float max_diff_nope = 0;
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            const float *a = Q + ((int64_t)s * heads + h) * hd;
            const float *b = Qref + ((int64_t)s * heads + h) * hd;
            for (int j = rotary_dim; j < hd; j++) {
                float d = fabsf(a[j] - b[j]);
                if (d > max_diff_nope) max_diff_nope = d;
            }
        }
    }

    fprintf(stderr, "   rope: max abs diff = %.2e (tol 1e-5), theta=5e6, rotary_dim=%d, nope-half unmodified (delta=%.2e)\n",
            max_diff_rot, rotary_dim, max_diff_nope);
    CHECK(max_diff_rot < 1e-5, "partial RoPE rotated half matches reference within 1e-5");
    CHECK(max_diff_nope == 0.0f, "partial RoPE nope half (dims 64-127) is unmodified");
    (void)heads;

    /* Spot-check rotation direction at pos=1, head=0, j=0:
     *   inv = theta^0 = 1, ang = 1, x[0] = x0*cos(1) - x1*sin(1).
     * Verify the sign matches HF (NOT the inverted-sign convention
     * x[0] = x0*cos + x1*sin which would be wrong). */
    {
        float v[4] = {1.0f, 0.5f, 0.0f, 0.0f};
        float v_ref[4] = {1.0f, 0.5f, 0.0f, 0.0f};
        rope(v, 1, theta, 4, 4);          /* rotary_dim=4 to exercise both halves */
        ref_rope(v_ref, 1, (double)theta, 4, 4);
        CHECK(fabsf(v[0] - v_ref[0]) < 1e-6, "rotation direction matches HF rotate_half convention");
        CHECK(fabsf(v[1] - v_ref[1]) < 1e-6, "rotation direction (high half) matches HF convention");
    }

    free(Q);
    free(Qref);
}

int main(void) {
    fprintf(stderr, "=== rope unit tests ===\n");
    test_partial_rope();
    if (g_fail) {
        fprintf(stderr, "\nFAILED: one or more rope tests failed\n");
        return 1;
    }
    fprintf(stderr, "\nPASS: all rope tests passed\n");
    return 0;
}
