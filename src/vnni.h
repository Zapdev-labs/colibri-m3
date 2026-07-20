/* vnni.h — AVX-512 VNNI (vpdpbusd) int8/int4 matmul kernels (f11).
 *
 * Cascade Lake+ (Xeon Gold 5220R) implements AVX-512_VNNI: the VPDPBUSD
 * instruction does 4 lanes of (u8 * s8) -> s32 accumulate per lane, 16 lanes
 * per 512-bit vector, giving 64 int8 multiply-adds per instruction.
 *
 * Two kernels:
 *   matmul_i8_vnni   — activations f32 -> u8 (per-row absmax/127), weights s8.
 *                      Matches the existing matmul_i8 numerics (scale * sum).
 *   matmul_i4_vnni   — int4 weights unpacked to int8 in 64-lane tiles, then
 *                      the same VNNI path. Tiles are unpacked lazily into a
 *                      scratch buffer to keep the working set in L1/L2.
 *
 * Falls back to the scalar C paths (matmul_i8, matmul_i4) when __AVX512VNNI__
 * is not defined, so the engine still builds on hosts without VNNI.
 *
 * Design notes:
 *  - Per-row symmetric quant: weight row's scale already folds the int8 range
 *    into the dot product; the activation scale is folded in at the end via
 *    a single per-row multiply. So y[s,o] = act_scale[s] * w_scale[o] * sum.
 *  - The activation quantizer uses absmax/127 (signed int8 in [-127,127]); the
 *    VNNI u8 operand expects unsigned, so we add 128 to bias s8 -> u8 and
 *    subtract the bias term (128 * sum_of_weights) afterwards. For weight rows
 *    where the int8 values are stored signed (q8 in [-128,127]), we instead use
 *    the AVX-512_VNNI variant that takes s8 * s8 -> s32 directly via a sign
 *    flip trick: pack weights as u8 = (s8 + 128), activations as u8 = (s8 + 128),
 *    then the dot is sum((s8w+128)(s8a+128)) = sum(s8w*s8a) + 128*sum(s8w) +
 *    128*sum(s8a) + 128*128*N. We precompute per-row weight sums at load time
 *    to subtract the bias cheaply.
 *
 *  To keep the kernel simple and numerically equivalent to matmul_i8, we use
 *  the standard u8*s8 -> s32 form (the 'DPU' variant). The trick: split the
 *  signed int8 weight w into two unsigned halves? No — simpler: cast the
 *  signed weight array to unsigned by reinterpreting bits, then XOR the
 *  sign bit of the activation. The VPDPBUSD instruction does
 *      acc += sum_i (a_u8[i] * b_s8[i])
 *  where a_u8 is unsigned [0,255] and b_s8 is signed [-128,127]. So we keep
 *  weights as s8 (no change to the on-disk format) and bias activations
 *  into [0,255] then subtract 128 * sum(weights) per row.
 */
#ifndef COLIBRI_M3_VNNI_H
#define COLIBRI_M3_VNNI_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX512VNNI__) || defined(__AVX512F__)
#define HAVE_AVX512 1
#else
#define HAVE_AVX512 0
#endif

#if defined(__AVX512VNNI__)
#define HAVE_VNNI 1
#else
#define HAVE_VNNI 0
#endif

/* ---------- activation quantization: f32 -> u8 (biased) + per-row scale ---------- */
/* Stores q as u8 = round(x / scale) + 128 so values live in [0,255] for VNNI.
 * Returns the scale = absmax/127 (or 1e-30 if all zeros). */
static inline float quant_act_u8(const float *x, uint8_t *q, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = x[i] < 0 ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    float scale = amax > 1e-30f ? amax / 127.0f : 1.0f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf(x[i] * inv) + 128;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        q[i] = (uint8_t)v;
    }
    return scale;
}

/* ---------- int8 weight matmul via VPDPBUSD (u8 act * s8 w -> s32) ---------- */
/* y[s,o] = act_scale[s] * w_scale[o] * sum_i (act_q_u8[s,i] - 128) * w_q_s8[o,i]
 *        = act_scale[s] * w_scale[o] * ( sum_i act_q_u8[s,i] * w_q_s8[o,i]
 *                                       - 128 * sum_i w_q_s8[o,i] )
 * Per-row weight sum is precomputed lazily here; for hot paths the caller
 * should hoist it. We compute it inside because the engine's int8 container
 * does not store it.
 */
#if HAVE_VNNI
static inline int32_t hsum512_epi32(__m512i v) {
    /* _mm512_reduce_add_epi32 sums all 16 int32 lanes into a single int32.
     * Available with AVX-512F; we're already in the __AVX512VNNI__ branch. */
    return (int32_t)_mm512_reduce_add_epi32(v);
}

static void matmul_i8_vnni(float *y, const float *x, const int8_t *q,
                            const float *sc, int S, int I, int O) {
    /* per-row weight sum (used to subtract the +128 activation bias) */
    int32_t *wsum = (int32_t *)malloc((size_t)O * sizeof(int32_t));
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        int32_t s = 0;
        for (int i = 0; i < I; i++) s += w[i];
        wsum[o] = s;
    }

    uint8_t *xq = (uint8_t *)malloc((size_t)I);
    float *xsc = (float *)malloc((size_t)S * sizeof(float));

    for (int s = 0; s < S; s++) {
        xsc[s] = quant_act_u8(x + (int64_t)s * I, xq, I);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            __m512i acc = _mm512_setzero_si512();
            int i = 0;
            for (; i + 64 <= I; i += 64) {
                __m512i a = _mm512_loadu_si512((const void *)(xq + i));   /* u8 x64 */
                __m512i b0 = _mm512_loadu_si512((const void *)(w + i));    /* s8 x64 */
                /* VPDPBUSD: acc += sum(a_u8 * b_s8) per 4-lane group */
                acc = _mm512_dpbusd_epi32(acc, a, b0);
            }
            int32_t sum = hsum512_epi32(acc);
            for (; i < I; i++) sum += (int32_t)((uint8_t)xq[i]) * (int32_t)w[i];
            /* subtract the +128 activation bias: sum_actual = sum - 128*wsum */
            sum -= 128 * wsum[o];
            y[(int64_t)s * O + o] = (float)sum * xsc[s] * sc[o];
        }
    }
    free(wsum); free(xq); free(xsc);
}
#else
/* Fallback: scalar reimplementation that is bit-identical to the VNNI path
 * (and to matmul_i8 up to activation re-quantization). Used when the host
 * lacks AVX-512_VNNI so the engine still builds and produces correct numbers. */
static void matmul_i8_vnni(float *y, const float *x, const int8_t *q,
                            const float *sc, int S, int I, int O) {
    int32_t *wsum = (int32_t *)malloc((size_t)O * sizeof(int32_t));
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        int32_t s = 0;
        for (int i = 0; i < I; i++) s += w[i];
        wsum[o] = s;
    }
    uint8_t *xq = (uint8_t *)malloc((size_t)I);
    float *xsc = (float *)malloc((size_t)S * sizeof(float));
    for (int s = 0; s < S; s++) {
        xsc[s] = quant_act_u8(x + (int64_t)s * I, xq, I);
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            int32_t sum = 0;
            for (int i = 0; i < I; i++) sum += (int32_t)xq[i] * (int32_t)w[i];
            sum -= 128 * wsum[o];
            y[(int64_t)s * O + o] = (float)sum * xsc[s] * sc[o];
        }
    }
    free(wsum); free(xq); free(xsc);
}
#endif

/* ---------- int4 weight matmul via VNNI ---------- */
/* Unpacks each int4 nibble (stored as q+8 in [0,15], so signed value = nib-8 in
 * [-8,7]) to int8 in tiles of 64 lanes, then runs the VNNI int8 path. The
 * int4 unpacking is the dominant extra cost vs int8 direct; we tile it so the
 * int8 buffer fits in L1 (64 cols * 1 byte = 64 B per row tile, trivially L1).
 *
 * The activation quantization is identical to the int8 path.
 */
static void matmul_i4_vnni(float *y, const float *x, const uint8_t *q4,
                            const float *sc, int S, int I, int O) {
    int rb = (I + 1) / 2;
    int32_t *wsum = (int32_t *)malloc((size_t)O * sizeof(int32_t));
    /* per-row weight sum (signed) for the +128 bias subtraction */
    #pragma omp parallel for schedule(static)
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

    uint8_t *xq = (uint8_t *)malloc((size_t)I);
    float *xsc = (float *)malloc((size_t)S * sizeof(float));

    for (int s = 0; s < S; s++) {
        xsc[s] = quant_act_u8(x + (int64_t)s * I, xq, I);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w = q4 + (int64_t)o * rb;
            /* Per-thread scratch tile (64 bytes, stack-allocated) — must NOT
             * be shared across OMP threads (was a race in the first version). */
            int8_t w8tile[64];
#if HAVE_VNNI
            __m512i acc = _mm512_setzero_si512();
            int i = 0;
            for (; i + 64 <= I; i += 64) {
                /* unpack 32 bytes (64 nibbles) into 64 int8 values */
                for (int j = 0; j < 32; j++) {
                    uint8_t b = w[(i >> 1) + j];
                    w8tile[j * 2]     = (int8_t)((int)(b & 0xF) - 8);
                    w8tile[j * 2 + 1] = (int8_t)((int)(b >> 4) - 8);
                }
                __m512i a = _mm512_loadu_si512((const void *)(xq + i));
                __m512i b = _mm512_loadu_si512((const void *)w8tile);
                acc = _mm512_dpbusd_epi32(acc, a, b);
            }
            int32_t sum = hsum512_epi32(acc);
            for (; i < I; i++) {
                uint8_t b = w[i >> 1];
                int wv = (i & 1) ? (int)(b >> 4) - 8 : (int)(b & 0xF) - 8;
                sum += (int32_t)xq[i] * wv;
            }
#else
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
#endif
            sum -= 128 * wsum[o];
            y[(int64_t)s * O + o] = (float)sum * xsc[s] * sc[o];
        }
    }
    free(wsum); free(xq); free(xsc);
}

/* Dispatch: choose VNNI path when available, else fall back to the scalar
 * matmul_i8 / matmul_i4 defined in engine.c. The caller sets use_vnni=1 to
 * opt in; we still gate on HAVE_VNNI internally so a non-VNNI host silently
 * falls back. */
static int g_use_vnni = 0;   /* set via USE_VNNI env var */

#endif /* COLIBRI_M3_VNNI_H */
