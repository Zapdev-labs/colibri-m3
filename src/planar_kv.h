/* PLANAR KV: block-diagonal 2D Givens rotation + low-bit storage.
 * Inspired by PlanarQuant/RotorQuant (scrya-com/rotorquant) — CPU path.
 * Fixed pi/4 rotations on (0,1),(2,3),... pairs; no trained rotors. Practical
 * CPU approximation that spreads KV magnitude before per-dim scalar quant. */
#ifndef COLI_PLANAR_KV_H
#define COLI_PLANAR_KV_H
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
  uint8_t *data;
  float scale;
} PlanarKVRow;

static const float PLANAR_C = 0.7071067811865476f; /* cos(pi/4) */
static const float PLANAR_S = 0.7071067811865476f; /* sin(pi/4) */

static inline int planar_qmax(int bits) {
  if (bits == 3) return 3;
  if (bits == 4) return 7;
  return 7;
}

static inline void planar_givens_forward(float *x, int hd) {
  for (int i = 0; i + 1 < hd; i += 2) {
    float x0 = x[i], x1 = x[i + 1];
    x[i]     = PLANAR_C * x0 - PLANAR_S * x1;
    x[i + 1] = PLANAR_S * x0 + PLANAR_C * x1;
  }
}

static inline void planar_givens_inverse(const float *y, float *x, int hd) {
  for (int i = 0; i + 1 < hd; i += 2) {
    float y0 = y[i], y1 = y[i + 1];
    x[i]     = PLANAR_C * y0 + PLANAR_S * y1;
    x[i + 1] = -PLANAR_S * y0 + PLANAR_C * y1;
  }
}

static inline float planar_kv_encode(const float *x, int8_t *codes, int hd, int bits) {
  float tmp[4096];
  float *buf = (hd <= 4096) ? tmp : (float *)malloc((size_t)hd * sizeof(float));
  if (!buf) return 1.f;
  memcpy(buf, x, (size_t)hd * sizeof(float));
  planar_givens_forward(buf, hd);
  int qmax = planar_qmax(bits);
  int qmin = -(qmax + 1);
  float amax = 0.f;
  for (int i = 0; i < hd; i++) {
    float a = fabsf(buf[i]);
    if (a > amax) amax = a;
  }
  float scale = (amax > 1e-8f) ? (amax / (float)qmax) : 1.f;
  float inv = 1.f / scale;
  for (int i = 0; i < hd; i++) {
    float q = roundf(buf[i] * inv);
    if (q < (float)qmin) q = (float)qmin;
    if (q > (float)qmax) q = (float)qmax;
    codes[i] = (int8_t)q;
  }
  if (buf != tmp) free(buf);
  return scale;
}

static inline void planar_kv_decode(float *out, const int8_t *codes, float scale, int hd, int bits) {
  (void)bits;
  float rot[4096];
  float *mid = (hd <= 4096) ? rot : (float *)malloc((size_t)hd * sizeof(float));
  if (!mid) { memset(out, 0, (size_t)hd * sizeof(float)); return; }
  for (int i = 0; i < hd; i++) mid[i] = (float)codes[i] * scale;
  planar_givens_inverse(mid, out, hd);
  if (mid != rot) free(mid);
}

static inline float planar_kv_dot_q(const float *q, const int8_t *codes, float scale, int hd, int bits) {
  (void)bits;
  float s = 0.f;
  for (int i = 0; i + 1 < hd; i += 2) {
    float q0 = q[i], q1 = q[i + 1];
    float rq0 = PLANAR_C * q0 - PLANAR_S * q1;
    float rq1 = PLANAR_S * q0 + PLANAR_C * q1;
    s += rq0 * (float)codes[i] + rq1 * (float)codes[i + 1];
  }
  if (hd & 1)
    s += q[hd - 1] * (float)codes[hd - 1];
  return s * scale;
}

#endif
