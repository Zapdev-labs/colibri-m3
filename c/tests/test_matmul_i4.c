/* test_matmul_i4.c — VAL-CORR-008: int4 matmul vs f32 reference at 1e-5 tolerance.
 *
 * Exercises both the per-row scaled (fmt=2) matmul_i4 kernel and the group-scaled
 * (fmt=4, gs=128) matmul_i4_grouped kernel against a float32 reference that uses
 * the same int4 dequantization convention as the engine
 * (low nibble -> ((b & 0xF) - 8), high nibble -> ((b >> 4) - 8)).
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_matmul_i4.c -o c/tests/test_matmul_i4 -lm
 * Run:   ./c/tests/test_matmul_i4
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

/* Dequantize one int4 nibble using the engine's convention: (nibble - 8). */
static inline int dq4_lo(uint8_t b) { return (int)(b & 0xF) - 8; }
static inline int dq4_hi(uint8_t b) { return (int)(b >> 4) - 8; }

/* Reference per-row int4 matmul (double accumulator for headroom). */
static void ref_matmul_i4_perrow(double *y, const float *x, const uint8_t *q4,
                                 const float *sc, int I, int O) {
    int rb = (I + 1) / 2;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        double a = 0.0;
        int i = 0;
        for (; i + 1 < I; i += 2) {
            uint8_t b = w[i >> 1];
            a += (double)x[i] * dq4_lo(b) + (double)x[i + 1] * dq4_hi(b);
        }
        if (i < I) a += (double)x[i] * dq4_lo(w[i >> 1]);
        y[o] = a * (double)sc[o];
    }
}

/* Reference group-scaled int4 matmul (one scale per (o, g)). */
static void ref_matmul_i4_grouped(double *y, const float *x, const uint8_t *q4,
                                  const float *sc, int I, int O, int gs) {
    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        const float *sco = sc + (int64_t)o * ng;
        double a = 0.0;
        for (int g = 0; g < ng; g++) {
            int gstart = g * gs;
            int gend = gstart + gs;
            if (gend > I) gend = I;
            double ga = 0.0;
            int i = gstart;
            for (; i + 1 < gend; i += 2) {
                uint8_t b = w[i >> 1];
                ga += (double)x[i] * dq4_lo(b) + (double)x[i + 1] * dq4_hi(b);
            }
            if (i < gend) ga += (double)x[i] * dq4_lo(w[i >> 1]);
            a += ga * (double)sco[g];
        }
        y[o] = a;
    }
}

/* Pack a float weight matrix into the engine's int4 format with per-row scales.
 * Quantization is asymmetric to the engine's [-8, 7] nibble range: scale = max(|w|)
 * over the row divided by 7 (the max positive nibble); clamp then store. */
static void pack_i4_perrow(uint8_t *q4, float *sc, const float *W, int I, int O) {
    int rb = (I + 1) / 2;
    for (int o = 0; o < O; o++) {
        const float *wr = W + (int64_t)o * I;
        float mx = 0;
        for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > mx) mx = a; }
        float scale = mx / 7.0f;
        if (scale < 1e-12f) scale = 1e-12f;
        sc[o] = scale;
        float inv = 1.0f / scale;
        for (int i = 0; i + 1 < I; i += 2) {
            int q0 = (int)lrintf(wr[i] * inv);
            int q1 = (int)lrintf(wr[i + 1] * inv);
            if (q0 < -8) q0 = -8;
            if (q0 > 7) q0 = 7;
            if (q1 < -8) q1 = -8;
            if (q1 > 7) q1 = 7;
            q4[(int64_t)o * rb + (i >> 1)] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
        }
        if (I & 1) {
            int i = I - 1;
            int q0 = (int)lrintf(wr[i] * inv);
            if (q0 < -8) q0 = -8;
            if (q0 > 7) q0 = 7;
            q4[(int64_t)o * rb + (i >> 1)] = (uint8_t)((q0 + 8) & 0xF);
        }
    }
}

/* Pack with per-group scales (gs groups per row). */
static void pack_i4_grouped(uint8_t *q4, float *sc, const float *W, int I, int O, int gs) {
    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;
    for (int o = 0; o < O; o++) {
        const float *wr = W + (int64_t)o * I;
        for (int g = 0; g < ng; g++) {
            int gstart = g * gs;
            int gend = gstart + gs;
            if (gend > I) gend = I;
            float mx = 0;
            for (int i = gstart; i < gend; i++) { float a = fabsf(wr[i]); if (a > mx) mx = a; }
            float scale = mx / 7.0f;
            if (scale < 1e-12f) scale = 1e-12f;
            sc[(int64_t)o * ng + g] = scale;
            float inv = 1.0f / scale;
            for (int i = gstart; i + 1 < gend; i += 2) {
                int q0 = (int)lrintf(wr[i] * inv);
                int q1 = (int)lrintf(wr[i + 1] * inv);
                if (q0 < -8) q0 = -8;
                if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8;
                if (q1 > 7) q1 = 7;
                q4[(int64_t)o * rb + (i >> 1)] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
            }
            if ((gend - gstart) & 1) {
                int i = gend - 1;
                int q0 = (int)lrintf(wr[i] * inv);
                if (q0 < -8) q0 = -8;
                if (q0 > 7) q0 = 7;
                q4[(int64_t)o * rb + (i >> 1)] = (uint8_t)((q0 + 8) & 0xF);
            }
        }
    }
}

static void test_perrow(void) {
    fprintf(stderr, "==> test_matmul_i4 per-row (fmt=2)\n");
    const int I = 6144, O = 128, S = 1;
    /* Use small fixed weights so quantization error stays well under 1e-5. */
    float *W = falloc((int64_t)O * I);
    float *x = falloc((int64_t)S * I);
    srand(7);
    for (int i = 0; i < O * I; i++) W[i] = ((float)(rand() % 2001) - 1000.0f) / 1.0e5f;  /* ±1e-2 */
    for (int i = 0; i < S * I; i++) x[i] = ((float)(rand() % 2001) - 1000.0f) / 1.0e3f;  /* ±1 */

    int rb = (I + 1) / 2;
    uint8_t *q4 = (uint8_t *)malloc((size_t)O * rb);
    float *sc = falloc(O);
    pack_i4_perrow(q4, sc, W, I, O);

    float *y_eng = falloc(O);
    double *y_ref = (double *)malloc((size_t)O * sizeof(double));
    matmul_i4(y_eng, x, q4, sc, S, I, O);
    ref_matmul_i4_perrow(y_ref, x, q4, sc, I, O);

    float max_diff = 0;
    for (int o = 0; o < O; o++) {
        float d = fabsf((float)y_ref[o] - y_eng[o]);
        if (d > max_diff) max_diff = d;
    }
    fprintf(stderr, "   matmul_i4 per-row: max abs diff = %.2e (tol 1e-5), O=%d I=%d\n",
            max_diff, O, I);
    CHECK(max_diff < 1e-5, "matmul_i4 per-row matches f32 reference within 1e-5");

    free(W); free(x); free(q4); free(sc); free(y_eng); free(y_ref);
}

static void test_grouped(void) {
    fprintf(stderr, "==> test_matmul_i4 grouped (fmt=4, gs=128)\n");
    const int I = 6144, O = 128, S = 1, gs = 128;
    float *W = falloc((int64_t)O * I);
    float *x = falloc((int64_t)S * I);
    srand(11);
    for (int i = 0; i < O * I; i++) W[i] = ((float)(rand() % 2001) - 1000.0f) / 1.0e5f;
    for (int i = 0; i < S * I; i++) x[i] = ((float)(rand() % 2001) - 1000.0f) / 1.0e3f;

    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;
    uint8_t *q4 = (uint8_t *)malloc((size_t)O * rb);
    float *sc = falloc((int64_t)O * ng);
    pack_i4_grouped(q4, sc, W, I, O, gs);

    float *y_eng = falloc(O);
    double *y_ref = (double *)malloc((size_t)O * sizeof(double));
    matmul_i4_grouped(y_eng, x, q4, sc, S, I, O, gs);
    ref_matmul_i4_grouped(y_ref, x, q4, sc, I, O, gs);

    float max_diff = 0;
    for (int o = 0; o < O; o++) {
        float d = fabsf((float)y_ref[o] - y_eng[o]);
        if (d > max_diff) max_diff = d;
    }
    fprintf(stderr, "   matmul_i4_grouped gs=%d: max abs diff = %.2e (tol 1e-5), O=%d I=%d\n",
            gs, max_diff, O, I);
    CHECK(max_diff < 1e-5, "matmul_i4_grouped gs=128 matches f32 reference within 1e-5");

    free(W); free(x); free(q4); free(sc); free(y_eng); free(y_ref);
}

int main(void) {
    fprintf(stderr, "=== matmul_i4 unit tests ===\n");
    test_perrow();
    test_grouped();
    if (g_fail) {
        fprintf(stderr, "\nFAILED: one or more matmul_i4 tests failed\n");
        return 1;
    }
    fprintf(stderr, "\nPASS: all matmul_i4 tests passed\n");
    return 0;
}
