/*
 * src/cuda_backend.cu — CUDA kernel implementations for lfmc
 *
 * Build with: make CUDA=1
 * Requires: CUDA Toolkit, cuBLAS
 *
 * All public functions accept host pointers and handle H2D/D2H transfers
 * internally, making it easy to drop-in replace CPU paths.
 *
 * All public functions call cuda_backend_init() lazily if not yet initialized.
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

/*
 * Grow-on-demand scratch device buffers.
 * cuda_adam_step uses slots a/b/c/tmp for params/grad/m/v and d as norm_sq.
 * All other ops use a and b (or just a for in-place).
 */
static float *g_dev_a   = NULL;   /* matmul A / in-place / adam params */
static float *g_dev_b   = NULL;   /* matmul B / src      / adam grad   */
static float *g_dev_c   = NULL;   /* matmul C            / adam m      */
static float *g_dev_tmp = NULL;   /*                       adam v      */
static float *g_dev_d   = NULL;   /* adam norm_sq scalar               */
static size_t g_dev_a_sz   = 0;
static size_t g_dev_b_sz   = 0;
static size_t g_dev_c_sz   = 0;
static size_t g_dev_tmp_sz = 0;
static size_t g_dev_d_sz   = 0;

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
        *buf = NULL;
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
    /* Free scratch buffers and reset pointers + sizes to prevent use-after-free
     * if the backend is re-initialized later. */
    if (g_dev_a)   { CUDA_CHECK(cudaFree(g_dev_a));   g_dev_a   = NULL; g_dev_a_sz   = 0; }
    if (g_dev_b)   { CUDA_CHECK(cudaFree(g_dev_b));   g_dev_b   = NULL; g_dev_b_sz   = 0; }
    if (g_dev_c)   { CUDA_CHECK(cudaFree(g_dev_c));   g_dev_c   = NULL; g_dev_c_sz   = 0; }
    if (g_dev_tmp) { CUDA_CHECK(cudaFree(g_dev_tmp)); g_dev_tmp = NULL; g_dev_tmp_sz = 0; }
    if (g_dev_d)   { CUDA_CHECK(cudaFree(g_dev_d));   g_dev_d   = NULL; g_dev_d_sz   = 0; }
    CUBLAS_CHECK(cublasDestroy(g_cublas));
    g_cublas = NULL;
    g_initialized = 0;
}

/* Lazy init helper — called at the top of every public entrypoint */
static inline void ensure_init(void) {
    if (!g_initialized) cuda_backend_init();
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
 *   cublasSgemm(OP_N, OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
 */
extern "C" void cuda_matmul(float *out, const float *a, const float *b,
                             int M, int K, int N) {
    ensure_init();
    size_t a_bytes  = (size_t)M * K * sizeof(float);
    size_t b_bytes  = (size_t)K * N * sizeof(float);
    size_t c_bytes  = (size_t)M * N * sizeof(float);

    float *da = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, a_bytes);
    float *db = ensure_dev_buf(&g_dev_b, &g_dev_b_sz, b_bytes);
    float *dc = ensure_dev_buf(&g_dev_c, &g_dev_c_sz, c_bytes);

    CUDA_CHECK(cudaMemcpy(da, a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, b_bytes, cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
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
    ensure_init();
    size_t bytes = (size_t)n * sizeof(float);
    float *d_dst = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    float *d_src = ensure_dev_buf(&g_dev_b, &g_dev_b_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_dst, dst, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
    k_saxpy<<<GRID(n), BLOCK>>>(d_dst, d_src, alpha, n);
    CUDA_CHECK(cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_add(float *dst, const float *src, int n) {
    ensure_init();
    size_t bytes = (size_t)n * sizeof(float);
    float *d_dst = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    float *d_src = ensure_dev_buf(&g_dev_b, &g_dev_b_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_dst, dst, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
    k_add<<<GRID(n), BLOCK>>>(d_dst, d_src, n);
    CUDA_CHECK(cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_scale(float *arr, float s, int n) {
    ensure_init();
    size_t bytes = (size_t)n * sizeof(float);
    float *d_arr = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice));
    k_scale<<<GRID(n), BLOCK>>>(d_arr, s, n);
    CUDA_CHECK(cudaMemcpy(arr, d_arr, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_tanh_activation(float *arr, int n) {
    ensure_init();
    size_t bytes = (size_t)n * sizeof(float);
    float *d_arr = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice));
    k_tanh<<<GRID(n), BLOCK>>>(d_arr, n);
    CUDA_CHECK(cudaMemcpy(arr, d_arr, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_silu_activation(float *arr, int n) {
    ensure_init();
    size_t bytes = (size_t)n * sizeof(float);
    float *d_arr = ensure_dev_buf(&g_dev_a, &g_dev_a_sz, bytes);
    CUDA_CHECK(cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice));
    k_silu<<<GRID(n), BLOCK>>>(d_arr, n);
    CUDA_CHECK(cudaMemcpy(arr, d_arr, bytes, cudaMemcpyDeviceToHost));
}

/* ── Fused Adam kernel ───────────────────────────────────────────────────── */
/*
 * Two-pass implementation using the shared grow-on-demand scratch buffers:
 *   g_dev_a  = d_params
 *   g_dev_b  = d_grad
 *   g_dev_c  = d_m
 *   g_dev_tmp= d_v
 *   g_dev_d  = d_norm_sq  (single float)
 *
 * No per-call cudaMalloc — buffers are grown once and reused across steps.
 *
 * Pass 1 (k_adam_norm): reduce sum(grad[i]^2) via atomicAdd into d_norm_sq
 * Pass 2 (k_adam_update): clip + m/v update + param update
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
    ensure_init();
    size_t bytes = (size_t)n * sizeof(float);

    /* Reuse shared scratch buffers — no per-call allocations */
    float *d_params  = ensure_dev_buf(&g_dev_a,   &g_dev_a_sz,   bytes);
    float *d_grad    = ensure_dev_buf(&g_dev_b,   &g_dev_b_sz,   bytes);
    float *d_m       = ensure_dev_buf(&g_dev_c,   &g_dev_c_sz,   bytes);
    float *d_v       = ensure_dev_buf(&g_dev_tmp, &g_dev_tmp_sz, bytes);
    float *d_norm_sq = ensure_dev_buf(&g_dev_d,   &g_dev_d_sz,   sizeof(float));

    CUDA_CHECK(cudaMemcpy(d_params, params, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad,   grad,   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m,      m,      bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,      v,      bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_norm_sq, 0, sizeof(float)));

    /* Pass 1: compute gradient L2 norm^2 */
    k_adam_norm<<<GRID(n), BLOCK>>>(d_grad, d_norm_sq, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float norm_sq_h;
    CUDA_CHECK(cudaMemcpy(&norm_sq_h, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost));
    float norm = sqrtf(norm_sq_h);
    float clip_scale = (norm > clip) ? clip / norm : 1.0f;

    /* Bias-corrected learning rate */
    float lr_t = lr * sqrtf(1.0f - powf(b2, (float)step))
                    / (1.0f - powf(b1, (float)step));

    /* Pass 2: clip + update m, v, params */
    k_adam_update<<<GRID(n), BLOCK>>>(d_params, d_grad, d_m, d_v,
                                       n, clip_scale, lr_t, b1, b2, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(params, d_params, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m,      d_m,      bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v,      d_v,      bytes, cudaMemcpyDeviceToHost));
}

#endif /* USE_CUDA */
