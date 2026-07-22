/* test_int8_kv.c — f14: INT8 KV cache correctness for long-context decode.
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_int8_kv.c -o c/tests/test_int8_kv -lm
 */
#ifndef TESTING
#define TESTING
#endif
#include "../src/engine.c"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float frand(void) { return (float)((rand() % 2000) - 1000) / 1000.0f; }

static int test_roundtrip(int hd) {
    float *x = malloc((size_t)hd * sizeof(float));
    int8_t *q = malloc((size_t)hd);
    float *y = malloc((size_t)hd * sizeof(float));
    for (int i = 0; i < hd; i++) x[i] = frand();
    float scale = kv_quantize_i8(x, q, hd);
    kv_dequantize_i8(y, q, scale, hd);
    float max_diff = 0, max_abs = 0;
    for (int i = 0; i < hd; i++) {
        float d = fabsf(x[i] - y[i]);
        if (d > max_diff) max_diff = d;
        if (fabsf(x[i]) > max_abs) max_abs = fabsf(x[i]);
    }
    float rel = max_abs > 1e-6 ? max_diff / max_abs : 0;
    printf("[test_int8_kv] hd=%d roundtrip max_diff=%.4f rel=%.4f %s\n",
           hd, max_diff, rel, rel < 0.02 ? "PASS" : "FAIL");
    free(x); free(q); free(y);
    return rel < 0.02 ? 0 : 1;
}

static int test_attention_dot(int n_pos) {
    int hd = 128;
    float *q = malloc((size_t)hd * sizeof(float));
    float *k = malloc((size_t)n_pos * hd * sizeof(float));
    int8_t *kq = malloc((size_t)n_pos * hd);
    float *ks = malloc((size_t)n_pos * sizeof(float));
    for (int i = 0; i < hd; i++) q[i] = frand() * 0.1f;
    for (int p = 0; p < n_pos; p++) {
        for (int i = 0; i < hd; i++) k[(int64_t)p * hd + i] = frand() * 0.1f;
        ks[p] = kv_quantize_i8(k + (int64_t)p * hd, kq + (int64_t)p * hd, hd);
    }
    float max_diff = 0;
    for (int p = 0; p < n_pos; p++) {
        float sf = 0, si = 0;
        for (int i = 0; i < hd; i++) {
            sf += q[i] * k[(int64_t)p * hd + i];
            si += q[i] * (float)kq[(int64_t)p * hd + i] * ks[p];
        }
        float d = fabsf(sf - si);
        if (d > max_diff) max_diff = d;
    }
    printf("[test_int8_kv] attn_dot n_pos=%d max_diff=%.6f %s\n",
           n_pos, max_diff, max_diff < 0.05 ? "PASS" : "FAIL");
    free(q); free(k); free(kq); free(ks);
    return max_diff < 0.05 ? 0 : 1;
}

static int test_long_context_budget(void) {
    long bytes_per_tok = 4L * 60 * 128 * 2;   /* 61440 */
    long ctx_32k = bytes_per_tok * 32768;
    long ctx_64k = bytes_per_tok * 65536;
    long ctx_1m = bytes_per_tok * 1048576L;
    long budget = 350L * 1024 * 1024 * 1024;
    printf("[test_int8_kv] budget: 32K=%ld MB, 64K=%ld MB, 1M=%ld MB\n",
           ctx_32k / (1024*1024), ctx_64k / (1024*1024), ctx_1m / (1024*1024));
    printf("[test_int8_kv] 350GB ceiling: 32K fits=%d, 64K fits=%d, 1M fits=%d\n",
           ctx_32k < budget, ctx_64k < budget, ctx_1m < budget);
    int rc = (ctx_32k < budget && ctx_64k < budget) ? 0 : 1;
    printf("[test_int8_kv] budget test %s\n", rc ? "FAIL" : "PASS");
    return rc;
}

int main(void) {
    srand(42);
    int rc = 0;
    rc |= test_roundtrip(128);
    rc |= test_roundtrip(64);
    rc |= test_attention_dot(8);
    rc |= test_attention_dot(1024);
    rc |= test_attention_dot(32768);
    rc |= test_long_context_budget();
    printf("[test_int8_kv] overall %s\n", rc ? "FAIL" : "PASS");
    return rc;
}
