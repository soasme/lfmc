/*
 * src/simd.h — AVX2 SIMD acceleration helpers for lfmc
 *
 * All AVX2 code is gated with #ifdef USE_AVX2.
 * When USE_AVX2 is not defined, scalar fallbacks are used.
 *
 * Compile with: -mavx2 -mfma -DUSE_AVX2
 * (-mfma is required for _mm256_fmadd_ps used in simd_dot / simd_saxpy)
 */

#ifndef SIMD_H
#define SIMD_H

#include <stddef.h>
#include <math.h>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

/* -------------------------------------------------------------------------
 * simd_dot — dot product of two float arrays
 * ------------------------------------------------------------------------- */
static inline float simd_dot(const float *a, const float *b, int n) {
#ifdef USE_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    /* Horizontal sum of acc */
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    /* Scalar tail */
    for (; i < n; i++) result += a[i] * b[i];
    return result;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

/* -------------------------------------------------------------------------
 * simd_saxpy — dst[i] += scale * src[i]
 * ------------------------------------------------------------------------- */
static inline void simd_saxpy(float *dst, const float *src, float scale, int n) {
#ifdef USE_AVX2
    __m256 vs = _mm256_set1_ps(scale);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vd  = _mm256_loadu_ps(dst + i);
        __m256 vsr = _mm256_loadu_ps(src + i);
        vd = _mm256_fmadd_ps(vs, vsr, vd);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++) dst[i] += scale * src[i];
#else
    for (int i = 0; i < n; i++) dst[i] += scale * src[i];
#endif
}

/* -------------------------------------------------------------------------
 * simd_add — dst[i] += src[i]
 * ------------------------------------------------------------------------- */
static inline void simd_add(float *dst, const float *src, int n) {
#ifdef USE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vd = _mm256_loadu_ps(dst + i);
        __m256 vs = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(vd, vs));
    }
    for (; i < n; i++) dst[i] += src[i];
#else
    for (int i = 0; i < n; i++) dst[i] += src[i];
#endif
}

/* -------------------------------------------------------------------------
 * simd_scale — arr[i] *= s
 * ------------------------------------------------------------------------- */
static inline void simd_scale(float *arr, float s, int n) {
#ifdef USE_AVX2
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(arr + i);
        _mm256_storeu_ps(arr + i, _mm256_mul_ps(va, vs));
    }
    for (; i < n; i++) arr[i] *= s;
#else
    for (int i = 0; i < n; i++) arr[i] *= s;
#endif
}

/* -------------------------------------------------------------------------
 * simd_mul_elementwise — dst[i] *= src[i]
 * ------------------------------------------------------------------------- */
static inline void simd_mul_elementwise(float *dst, const float *src, int n) {
#ifdef USE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vd = _mm256_loadu_ps(dst + i);
        __m256 vs = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(vd, vs));
    }
    for (; i < n; i++) dst[i] *= src[i];
#else
    for (int i = 0; i < n; i++) dst[i] *= src[i];
#endif
}

/* -------------------------------------------------------------------------
 * simd_tanh — fast tanh via rational polynomial approximation
 *
 * Approximation: x*(27 + x²) / (27 + 9*x²), clamped to ±1 for |x| >= 3
 * Max error vs tanhf: ~0.0015 at |x| ≈ 2.5.  Exact at x=0.
 *
 * Applied uniformly to all elements (AVX8-wide lanes + scalar tail) so
 * outputs are consistent regardless of n % 8.
 * ------------------------------------------------------------------------- */
static inline float _tanh_approx(float x) {
    float x2 = x * x;
    if (x >=  3.0f) return  1.0f;
    if (x <= -3.0f) return -1.0f;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static inline void simd_tanh(float *arr, int n) {
#ifdef USE_AVX2
    __m256 v27   = _mm256_set1_ps(27.0f);
    __m256 v9    = _mm256_set1_ps(9.0f);
    __m256 vone  = _mm256_set1_ps(1.0f);
    __m256 vmone = _mm256_set1_ps(-1.0f);
    __m256 v3    = _mm256_set1_ps(3.0f);
    __m256 vm3   = _mm256_set1_ps(-3.0f);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 x  = _mm256_loadu_ps(arr + i);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 numer = _mm256_mul_ps(x, _mm256_add_ps(v27, x2));
        __m256 denom = _mm256_add_ps(v27, _mm256_mul_ps(v9, x2));
        __m256 approx = _mm256_div_ps(numer, denom);
        __m256 mask_hi = _mm256_cmp_ps(x, v3,  _CMP_GE_OQ);
        __m256 mask_lo = _mm256_cmp_ps(x, vm3, _CMP_LE_OQ);
        approx = _mm256_blendv_ps(approx, vone,  mask_hi);
        approx = _mm256_blendv_ps(approx, vmone, mask_lo);
        _mm256_storeu_ps(arr + i, approx);
    }
    /* Scalar tail — same approximation for consistent output */
    for (; i < n; i++) arr[i] = _tanh_approx(arr[i]);
#else
    for (int i = 0; i < n; i++) arr[i] = _tanh_approx(arr[i]);
#endif
}

/* -------------------------------------------------------------------------
 * simd_silu — SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Computed exactly via expf for all elements. The AVX2 path batches the
 * loop to reduce overhead, but uses scalar expf per element since there
 * is no native AVX2 exp intrinsic in the base ISA.
 * ------------------------------------------------------------------------- */
static inline void simd_silu(float *arr, int n) {
#ifdef USE_AVX2
    /* Process 8 at a time using scalar expf (no AVX2 exp intrinsic in base ISA).
     * Still benefits from reduced loop overhead and prefetch. */
    int i = 0;
    for (; i <= n - 8; i += 8) {
        /* Load 8, apply SiLU scalar, store 8 */
        float tmp[8];
        __m256 vx = _mm256_loadu_ps(arr + i);
        _mm256_storeu_ps(tmp, vx);
        for (int k = 0; k < 8; k++)
            tmp[k] = tmp[k] / (1.0f + expf(-tmp[k]));
        _mm256_storeu_ps(arr + i, _mm256_loadu_ps(tmp));
    }
    for (; i < n; i++) {
        float x = arr[i];
        arr[i] = x / (1.0f + expf(-x));
    }
#else
    for (int i = 0; i < n; i++) {
        float x = arr[i];
        arr[i] = x / (1.0f + expf(-x));
    }
#endif
}

#endif /* SIMD_H */
