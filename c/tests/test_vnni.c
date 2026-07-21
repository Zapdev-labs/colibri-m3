/* test_vnni.c — f11: AVX-512 VNNI matmul kernel correctness.
 *
 * Compares matmul_i8_vnni / matmul_i4_vnni against a f32 reference that also
 * quantizes activations to int8 (matching the VNNI numerics bit-exactly), plus
 * a relaxed f32-reference check (the VNNI path introduces small activation
 * quantization error vs the engine's f32-activation matmul_i8/matmul_i4).
 *
 * Pass: bit-exact match against the act-quant reference; <5% relative error
 * vs the f32 reference.
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_vnni.c -o c/tests/test_vnni -lm
 */
#ifndef TESTING
#define TESTING
#endif
#include "../src/engine.c"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float frand(void) { return (float)((rand() % 2000) - 1000) / 1000.0f; }

/* Reference that mirrors the VNNI activation quantization exactly. */
static void matmul_i8_actq_ref(float *y, const float *x, const int8_t *q,
                                const float *sc, int S, int I, int O) {
    int32_t *wsum = malloc((size_t)O * sizeof(int32_t));
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        int32_t s = 0;
        for (int i = 0; i < I; i++) s += w[i];
        wsum[o] = s;
    }
    uint8_t *xq = malloc((size_t)I);
    for (int s = 0; s < S; s++) {
        float xs = quant_act_u8(x + (int64_t)s * I, xq, I);
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            int32_t sum = 0;
            for (int i = 0; i < I; i++) sum += (int32_t)xq[i] * (int32_t)w[i];
            sum -= 128 * wsum[o];
            y[(int64_t)s * O + o] = (float)sum * xs * sc[o];
        }
    }
    free(wsum); free(xq);
}

static void matmul_i4_actq_ref(float *y, const float *x, const uint8_t *q4,
                                const float *sc, int S, int I, int O) {
    int rb = (I + 1) / 2;
    int32_t *wsum = malloc((size_t)O * sizeof(int32_t));
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        int32_t s = 0;
        for (int i = 0; i + 1 < I; i += 2) {
            uint8_t b = w[i >> 1];
            s += (int)(b & 0xF) - 8;
            s += (int)(b >> 4) - 8;
        }
        if (I & 1) s += (int)(w[(I - 1) >> 1] & 0xF) - 8;
        wsum[o] = s;
    }
    uint8_t *xq = malloc((size_t)I);
    for (int s = 0; s < S; s++) {
        float xs = quant_act_u8(x + (int64_t)s * I, xq, I);
        for (int o = 0; o < O; o++) {
            const uint8_t *w = q4 + (int64_t)o * rb;
            int32_t sum = 0;
            for (int i = 0; i + 1 < I; i += 2) {
                uint8_t b = w[i >> 1];
                sum += (int32_t)xq[i] * ((int)(b & 0xF) - 8);
                sum += (int32_t)xq[i + 1] * ((int)(b >> 4) - 8);
            }
            if (I & 1) {
                uint8_t b = w[(I - 1) >> 1];
                sum += (int32_t)xq[I - 1] * ((int)(b & 0xF) - 8);
            }
            sum -= 128 * wsum[o];
            y[(int64_t)s * O + o] = (float)sum * xs * sc[o];
        }
    }
    free(wsum); free(xq);
}

static int test_i8(int S, int I, int O) {
    int8_t *q = malloc((size_t)O * I);
    float *sc = malloc((size_t)O * sizeof(float));
    float *x = malloc((size_t)S * I * sizeof(float));
    float *y_ref = malloc((size_t)S * O * sizeof(float));
    float *y_vnni = malloc((size_t)S * O * sizeof(float));
    for (int o = 0; o < O; o++) {
        sc[o] = 0.01f + frand() * 0.1f;
        for (int i = 0; i < I; i++)
            q[(int64_t)o * I + i] = (int8_t)(rand() % 256 - 128);
    }
    for (int64_t z = 0; z < (int64_t)S * I; z++) x[z] = frand();
    matmul_i8_actq_ref(y_ref, x, q, sc, S, I, O);
    matmul_i8_vnni(y_vnni, x, q, sc, S, I, O);
    float max_diff = 0, max_abs = 0;
    for (int64_t z = 0; z < (int64_t)S * O; z++) {
        float d = fabsf(y_ref[z] - y_vnni[z]);
        if (d > max_diff) max_diff = d;
        if (fabsf(y_ref[z]) > max_abs) max_abs = fabsf(y_ref[z]);
    }
    float rel = max_abs > 1e-6 ? max_diff / max_abs : 0;
    /* bit-exact expected (same numerics in scalar ref and VNNI) */
    int pass = rel < 1e-4;
    printf("[test_vnni] i8 S=%d I=%d O=%d: max_diff=%.6f rel=%.6f %s\n",
           S, I, O, max_diff, rel, pass ? "PASS" : "FAIL");
    free(q); free(sc); free(x); free(y_ref); free(y_vnni);
    return pass ? 0 : 1;
}

static int test_i4(int S, int I, int O) {
    int rb = (I + 1) / 2;
    uint8_t *q4 = malloc((size_t)O * rb);
    float *sc = malloc((size_t)O * sizeof(float));
    float *x = malloc((size_t)S * I * sizeof(float));
    float *y_ref = malloc((size_t)S * O * sizeof(float));
    float *y_vnni = malloc((size_t)S * O * sizeof(float));
    for (int o = 0; o < O; o++) {
        sc[o] = 0.01f + frand() * 0.1f;
        for (int i = 0; i < rb; i++)
            q4[(int64_t)o * rb + i] = (uint8_t)(rand() % 256);
    }
    for (int64_t z = 0; z < (int64_t)S * I; z++) x[z] = frand();
    matmul_i4_actq_ref(y_ref, x, q4, sc, S, I, O);
    matmul_i4_vnni(y_vnni, x, q4, sc, S, I, O);
    float max_diff = 0, max_abs = 0;
    for (int64_t z = 0; z < (int64_t)S * O; z++) {
        float d = fabsf(y_ref[z] - y_vnni[z]);
        if (d > max_diff) max_diff = d;
        if (fabsf(y_ref[z]) > max_abs) max_abs = fabsf(y_ref[z]);
    }
    float rel = max_abs > 1e-6 ? max_diff / max_abs : 0;
    int pass = rel < 1e-4;
    printf("[test_vnni] i4 S=%d I=%d O=%d: max_diff=%.6f rel=%.6f %s\n",
           S, I, O, max_diff, rel, pass ? "PASS" : "FAIL");
    free(q4); free(sc); free(x); free(y_ref); free(y_vnni);
    return pass ? 0 : 1;
}

/* Sanity: VNNI should be within ~5% of the engine's f32-activation matmul_i8
 * (the small drift is from activation quantization, expected and acceptable). */
static int test_i8_vs_f32(int S, int I, int O) {
    int8_t *q = malloc((size_t)O * I);
    float *sc = malloc((size_t)O * sizeof(float));
    float *x = malloc((size_t)S * I * sizeof(float));
    float *y_f32 = malloc((size_t)S * O * sizeof(float));
    float *y_vnni = malloc((size_t)S * O * sizeof(float));
    for (int o = 0; o < O; o++) {
        sc[o] = 0.01f + frand() * 0.1f;
        for (int i = 0; i < I; i++)
            q[(int64_t)o * I + i] = (int8_t)(rand() % 256 - 128);
    }
    for (int64_t z = 0; z < (int64_t)S * I; z++) x[z] = frand();
    matmul_i8(y_f32, x, q, sc, S, I, O);
    matmul_i8_vnni(y_vnni, x, q, sc, S, I, O);
    float max_diff = 0, max_abs = 0;
    for (int64_t z = 0; z < (int64_t)S * O; z++) {
        float d = fabsf(y_f32[z] - y_vnni[z]);
        if (d > max_diff) max_diff = d;
        if (fabsf(y_f32[z]) > max_abs) max_abs = fabsf(y_f32[z]);
    }
    float rel = max_abs > 1e-6 ? max_diff / max_abs : 0;
    int pass = rel < 0.05f;
    printf("[test_vnni] i8_vs_f32 S=%d I=%d O=%d: rel=%.4f %s\n",
           S, I, O, rel, pass ? "PASS" : "FAIL");
    free(q); free(sc); free(x); free(y_f32); free(y_vnni);
    return pass ? 0 : 1;
}

/* Compare matmul_i4_avx512 (the default decode path) vs scalar matmul_i4.
 * They should agree to within float32 rounding error (< 1e-4 relative). */
static int test_i4_avx512_vs_scalar(int S, int I, int O) {
    int rb = (I + 1) / 2;
    uint8_t *q4 = malloc((size_t)O * rb);
    float *sc = malloc((size_t)O * sizeof(float));
    float *x = malloc((size_t)S * I * sizeof(float));
    float *y_scalar = malloc((size_t)S * O * sizeof(float));
    float *y_avx512 = malloc((size_t)S * O * sizeof(float));
    for (int o = 0; o < O; o++) {
        sc[o] = 0.01f + frand() * 0.1f;
        for (int i = 0; i < rb; i++)
            q4[(int64_t)o * rb + i] = (uint8_t)(rand() % 256);
    }
    for (int64_t z = 0; z < (int64_t)S * I; z++) x[z] = frand();
    matmul_i4(y_scalar, x, q4, sc, S, I, O);
    matmul_i4_avx512(y_avx512, x, q4, sc, S, I, O);
    float max_diff = 0, max_abs = 0;
    for (int64_t z = 0; z < (int64_t)S * O; z++) {
        float d = fabsf(y_scalar[z] - y_avx512[z]);
        if (d > max_diff) max_diff = d;
        if (fabsf(y_scalar[z]) > max_abs) max_abs = fabsf(y_scalar[z]);
    }
    float rel = max_abs > 1e-6 ? max_diff / max_abs : 0;
    int pass = rel < 1e-4;
    printf("[test_vnni] i4_avx512_vs_scalar S=%d I=%d O=%d: max_diff=%.6f rel=%.6f %s\n",
           S, I, O, max_diff, rel, pass ? "PASS" : "FAIL");
    free(q4); free(sc); free(x); free(y_scalar); free(y_avx512);
    return pass ? 0 : 1;
}

int main(void) {
    srand(42);
    int rc = 0;
    rc |= test_i8(1, 64, 16);
    rc |= test_i8(1, 6144, 3072);
    rc |= test_i4(1, 64, 16);
    rc |= test_i4(1, 6144, 3072);
    rc |= test_i4(1, 6144, 8192);
    rc |= test_i8_vs_f32(1, 6144, 3072);
    rc |= test_i4_avx512_vs_scalar(1, 64, 16);
    rc |= test_i4_avx512_vs_scalar(1, 6144, 3072);
    rc |= test_i4_avx512_vs_scalar(1, 6144, 8192);
    rc |= test_i4_avx512_vs_scalar(1, 3072, 6144);
    printf("[test_vnni] overall %s\n", rc ? "FAIL" : "PASS");
    return rc;
}
