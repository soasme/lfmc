/*
 * src/cuda_backend.cu — CUDA kernel implementations for lfmc
 *
 * Build with: make CUDA=1
 * Requires: CUDA Toolkit, cuBLAS
 *
 * All public functions accept host pointers and handle H2D/D2H transfers
 * internally, making it easy to drop-in replace CPU paths.
 */

#include "cuda_backend.h"

#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* ── Global state ────────────────────────────────────────────────────────── */

static cublasHandle_t g_cublas = NULL;
static int g_initialized = 0;

/* Scratch device buffers — grown on demand */
static float *g_dev_a   = NULL;
static float *g_dev_b   = NULL;
static float *g_dev_c   = NULL;
static float *g_dev_tmp = NULL;   /* general scratch */
static size_t g_dev_a_sz = 0;
static size_t g_dev_b_sz = 0;
static size_t g_dev_c_sz = 0;
static size_t g_dev_tmp_sz = 0;

/* ── Error helpers ───────────────────────────────────────────────────────── */

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        exit(1); \
    } \
} while(0)

static float *ensure_dev_buf(float **buf, size_t *cur_sz, size_t need_sz) {
    if (need_sz > *cur_sz) {
        if (*buf) CUDA_CHECK(cudaFree(*buf));
        CUDA_CHECK(cudaMalloc((void**)buf, need_sz));
        *cur_sz = need_sz;
    }
    return *buf;
}

/* ── Init / free ─────────────────────────────────────────────────────────── */

extern "C" void cuda_backend_init(void) {
    if (g_initialized) return;
    CUBLAS_CHECK(cublasCreate(&g_cublas));
    g_initialized = 1;
    fprintf(stderr, "[cuda_backend] initialized\n");
}

extern "C" void cuda_backend_free(void) {
    if (!g_initialized) return;
    if (g_dev_a)   cudaFree(g_dev_a);
    if (g_dev_b)   cudaFree(g_dev_b);
    if (g_dev_c)   cudaFree(g_dev_c);
    if (g_dev_tmp) cudaFree(g_dev_tmp);
    cublasDestroy(g_cublas);
    g_cublas = NULL;
    g_initialized = 0;
}

/* ── cuda_matmul ─────────────────────────────────────────────────────────── */
/*
 * Row-major:  C[M,N] = A[M,K] @ B[K,N]
 *
 * cuBLAS uses column-major. The identity:
 *   row-major C = A @ B
 * is equivalent (in col-major cuBLAS convention) to:
 *   col-major C^T = B^T @ A^T
 * Since cuBLAS interprets our row-major A as col-major A^T etc., we call:
 *   cublasSgemm(N, N, C^T=B^T@A^T)  →  C = A@B  ✓
 *
 * cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
 * where m=N_out_cols, n=M_out_rows, k=K
 */
extern "C" void cuda_matmul(float *out, const float *a, const float *b,
                             int M, int K, int N) {
    size_t a_bytes  = (size_t)M * K * sizeof(float);
    size_t b_bytes  = (size_t)K * N * sizeof(float);
    size_t c_bytes  = (size_t)M * N * sizeof(float);

    float *da = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, a_bytes);
    float *db = ensure_dev_buf(&g_dev_b, &g_dev_b_sz, b_bytes);
    float *dc = ensure_dev_buf(&g_dev_c, &g_dev_c_sz, c_bytes);

    CUDA_CHECK(cudaMemcpy(da, a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, b_bytes, cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    /* C[M,N] = A[M,K] @ B[K,N]  row-major
     * = cublasSgemm(handle, OP_N, OP_N, N, M, K, &alpha, db, N, da, K, &beta, dc, N) */
    CUBLAS_CHECK(cublasSgemm(g_cublas,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, db, N, da, K,
                             &beta,  dc, N));

    CUDA_CHECK(cudaMemcpy(out, dc, c_bytes, cudaMemcpyDeviceToHost));
}

/* ── Element-wise kernels ────────────────────────────────────────────────── */

__global__ static void k_saxpy(float *dst, const float *src, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += alpha * src[i];
}

__global__ static void k_add(float *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ static void k_scale(float *arr, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] *= s;
}

__global__ static void k_tanh(float *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = tanhf(arr[i]);
}

__global__ static void k_silu(float *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = arr[i];
        arr[i] = x / (1.0f + expf(-x));
    }
}

#define BLOCK 256
#define GRID(n) (((n) + BLOCK - 1) / BLOCK)

extern "C" void cuda_saxpy(float *dst, const float *src, float alpha, int n) {
    size_t bytes = (size_t)n * sizeof(float);
    float *d_dst = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    float *d_src = ensure_dev_buf(&g_dev_b, &g_dev_b_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_dst, dst, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
    k_saxpy<<<GRID(n), BLOCK>>>(d_dst, d_src, alpha, n);
    CUDA_CHECK(cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_add(float *dst, const float *src, int n) {
    size_t bytes = (size_t)n * sizeof(float);
    float *d_dst = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    float *d_src = ensure_dev_buf(&g_dev_b, &g_dev_b_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_dst, dst, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
    k_add<<<GRID(n), BLOCK>>>(d_dst, d_src, n);
    CUDA_CHECK(cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_scale(float *arr, float s, int n) {
    size_t bytes = (size_t)n * sizeof(float);
    float *d_arr = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice));
    k_scale<<<GRID(n), BLOCK>>>(d_arr, s, n);
    CUDA_CHECK(cudaMemcpy(arr, d_arr, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_tanh_activation(float *arr, int n) {
    size_t bytes = (size_t)n * sizeof(float);
    float *d_arr = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice));
    k_tanh<<<GRID(n), BLOCK>>>(d_arr, n);
    CUDA_CHECK(cudaMemcpy(arr, d_arr, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_silu_activation(float *arr, int n) {
    size_t bytes = (size_t)n * sizeof(float);
    float *d_arr = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice));
    k_silu<<<GRID(n), BLOCK>>>(d_arr, n);
    CUDA_CHECK(cudaMemcpy(arr, d_arr, bytes, cudaMemcpyDeviceToHost));
}

/* ── Fused Adam kernel ───────────────────────────────────────────────────── */
/*
 * Two-pass implementation:
 *   Pass 1 (k_adam_norm): compute sum of squared gradients via atomicAdd
 *   Pass 2 (k_adam_update): clip + m/v update + param update
 */

__global__ static void k_adam_norm(const float *grad, float *norm_sq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(norm_sq, grad[i] * grad[i]);
}

__global__ static void k_adam_update(float *params, float *grad,
                                     float *m, float *v,
                                     int n, float clip_scale,
                                     float lr_t, float b1, float b2, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i] * clip_scale;
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        params[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps);
    }
}

extern "C" void cuda_adam_step(float *params, float *grad, float *m, float *v,
                                int n, int step,
                                float lr, float b1, float b2, float eps, float clip) {
    size_t bytes = (size_t)n * sizeof(float);

    /* Allocate device buffers */
    float *d_params, *d_grad, *d_m, *d_v, *d_norm_sq;
    CUDA_CHECK(cudaMalloc((void**)&d_params,  bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_grad,    bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_m,       bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_v,       bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_norm_sq, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_params, params, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad,   grad,   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m,      m,      bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,      v,      bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_norm_sq, 0, sizeof(float)));

    /* Pass 1: compute gradient norm^2 */
    k_adam_norm<<<GRID(n), BLOCK>>>(d_grad, d_norm_sq, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float norm_sq_h;
    CUDA_CHECK(cudaMemcpy(&norm_sq_h, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost));
    float norm = sqrtf(norm_sq_h);
    float clip_scale = (norm > clip) ? clip / norm : 1.0f;

    /* Bias-corrected learning rate */
    float lr_t = lr * sqrtf(1.0f - powf(b2, (float)step))
                    / (1.0f - powf(b1, (float)step));

    /* Pass 2: update */
    k_adam_update<<<GRID(n), BLOCK>>>(d_params, d_grad, d_m, d_v,
                                       n, clip_scale, lr_t, b1, b2, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(params, d_params, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m,      d_m,      bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v,      d_v,      bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_params); cudaFree(d_grad);
    cudaFree(d_m);      cudaFree(d_v);
    cudaFree(d_norm_sq);
}

#endif /* USE_CUDA */
