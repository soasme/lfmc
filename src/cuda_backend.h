/*
 * src/cuda_backend.h — CUDA acceleration backend for lfmc
 *
 * Compile with -DUSE_CUDA and link with -lcublas -lcudart.
 * Build with: make CUDA=1
 *
 * All declarations are guarded by #ifdef USE_CUDA so this header
 * is safe to include in non-CUDA builds.
 */

#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#ifdef USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize CUDA context, cuBLAS handle, and scratch buffers.
 * Must be called before any other cuda_* function. */
void cuda_backend_init(void);

/* Release all CUDA resources. */
void cuda_backend_free(void);

/* Matrix multiply: out[M,N] = a[M,K] @ b[K,N]  (row-major)
 * All pointers are host memory; function handles H2D/D2H transfers. */
void cuda_matmul(float *out, const float *a, const float *b,
                 int M, int K, int N);

/* dst[i] += alpha * src[i] */
void cuda_saxpy(float *dst, const float *src, float alpha, int n);

/* dst[i] += src[i] */
void cuda_add(float *dst, const float *src, int n);

/* arr[i] *= s */
void cuda_scale(float *arr, float s, int n);

/* arr[i] = tanh(arr[i]) */
void cuda_tanh_activation(float *arr, int n);

/* arr[i] = arr[i] * sigmoid(arr[i])  (SiLU) */
void cuda_silu_activation(float *arr, int n);

/* Fused Adam step with gradient clipping.
 * All arrays are host memory; the kernel handles transfers internally.
 *
 *   norm  = sqrt(sum(grad^2))
 *   if norm > clip: grad *= clip/norm
 *   step++
 *   lr_t = lr * sqrt(1 - b2^step) / (1 - b1^step)
 *   m[i]  = b1*m[i] + (1-b1)*grad[i]
 *   v[i]  = b2*v[i] + (1-b2)*grad[i]^2
 *   params[i] -= lr_t * m[i] / (sqrt(v[i]) + eps)
 */
void cuda_adam_step(float *params, float *grad, float *m, float *v,
                    int n, int step,
                    float lr, float b1, float b2, float eps, float clip);

#ifdef __cplusplus
}
#endif

#endif /* USE_CUDA */

#endif /* CUDA_BACKEND_H */
