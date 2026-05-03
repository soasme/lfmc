#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

/* -------------------------------------------------------------------------- */
/* Lifecycle                                                                   */
/* -------------------------------------------------------------------------- */

Tensor *tensor_alloc(int ndim, int *shape) {
    assert(ndim > 0 && ndim <= TENSOR_MAX_DIMS);
    Tensor *t = malloc(sizeof(Tensor));
    if (!t) { perror("tensor_alloc"); exit(1); }
    t->ndim = ndim;
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= (size_t)shape[i];
    }
    t->data = malloc(t->size * sizeof(float));
    if (!t->data) { perror("tensor_alloc data"); exit(1); }
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

Tensor *tensor_zeros(int ndim, int *shape) {
    Tensor *t = tensor_alloc(ndim, shape);
    memset(t->data, 0, t->size * sizeof(float));
    return t;
}

Tensor *tensor_ones(int ndim, int *shape) {
    Tensor *t = tensor_alloc(ndim, shape);
    tensor_fill(t, 1.0f);
    return t;
}

/* Box-Muller transform for Gaussian samples */
Tensor *tensor_randn(int ndim, int *shape, float std) {
    Tensor *t = tensor_alloc(ndim, shape);
    for (size_t i = 0; i + 1 < t->size; i += 2) {
        double u1 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 =  rand()        / ((double)RAND_MAX + 1.0);
        double r  = sqrt(-2.0 * log(u1));
        t->data[i]   = (float)(r * cos(2.0 * M_PI * u2) * std);
        t->data[i+1] = (float)(r * sin(2.0 * M_PI * u2) * std);
    }
    if (t->size % 2 == 1) {
        double u1 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 =  rand()        / ((double)RAND_MAX + 1.0);
        t->data[t->size-1] = (float)(sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2)*std);
    }
    return t;
}

/* -------------------------------------------------------------------------- */
/* Basic math                                                                  */
/* -------------------------------------------------------------------------- */

void tensor_fill(Tensor *t, float val) {
    for (size_t i = 0; i < t->size; i++) t->data[i] = val;
}

void tensor_scale(Tensor *t, float s) {
    for (size_t i = 0; i < t->size; i++) t->data[i] *= s;
}

void tensor_add_inplace(Tensor *dst, const Tensor *src) {
    assert(dst->size == src->size);
    for (size_t i = 0; i < dst->size; i++) dst->data[i] += src->data[i];
}

void tensor_mul_inplace(Tensor *dst, const Tensor *src) {
    assert(dst->size == src->size);
    for (size_t i = 0; i < dst->size; i++) dst->data[i] *= src->data[i];
}

void tensor_copy(Tensor *dst, const Tensor *src) {
    assert(dst->size == src->size);
    memcpy(dst->data, src->data, src->size * sizeof(float));
}

/* -------------------------------------------------------------------------- */
/* Neural-net ops                                                              */
/* -------------------------------------------------------------------------- */

/* Naive matmul: out[M,N] = a[M,K] @ b[K,N] */
void tensor_matmul(Tensor *out, const Tensor *a, const Tensor *b) {
    assert(a->ndim >= 2 && b->ndim >= 2 && out->ndim >= 2);
    int M = a->shape[a->ndim-2];
    int K = a->shape[a->ndim-1];
    int N = b->shape[b->ndim-1];
    assert(b->shape[b->ndim-2] == K);
    assert(out->shape[out->ndim-2] == M);
    assert(out->shape[out->ndim-1] == N);

    memset(out->data, 0, out->size * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float aval = a->data[m * K + k];
            for (int n = 0; n < N; n++)
                out->data[m * N + n] += aval * b->data[k * N + n];
        }
}

void tensor_softmax(Tensor *t, int axis) {
    /* Only 1-D softmax for now (axis ignored) */
    (void)axis;
    float max_val = t->data[0];
    for (size_t i = 1; i < t->size; i++)
        if (t->data[i] > max_val) max_val = t->data[i];
    float sum = 0.0f;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = expf(t->data[i] - max_val);
        sum += t->data[i];
    }
    for (size_t i = 0; i < t->size; i++) t->data[i] /= sum;
}

void tensor_layer_norm(Tensor *out, const Tensor *in,
                       const Tensor *w, const Tensor *b, float eps) {
    size_t n = in->size;
    float mean = 0.0f, var = 0.0f;
    for (size_t i = 0; i < n; i++) mean += in->data[i];
    mean /= (float)n;
    for (size_t i = 0; i < n; i++) {
        float d = in->data[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < n; i++)
        out->data[i] = (in->data[i] - mean) * inv_std * w->data[i] + b->data[i];
}

void tensor_silu(Tensor *t) {
    for (size_t i = 0; i < t->size; i++) {
        float x = t->data[i];
        t->data[i] = x / (1.0f + expf(-x));
    }
}

void tensor_tanh(Tensor *t) {
    for (size_t i = 0; i < t->size; i++) t->data[i] = tanhf(t->data[i]);
}

void tensor_sigmoid(Tensor *t) {
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
}

void tensor_relu(Tensor *t) {
    for (size_t i = 0; i < t->size; i++)
        if (t->data[i] < 0.0f) t->data[i] = 0.0f;
}

/* -------------------------------------------------------------------------- */
/* Reduction                                                                   */
/* -------------------------------------------------------------------------- */

float tensor_sum(const Tensor *t) {
    float s = 0.0f;
    for (size_t i = 0; i < t->size; i++) s += t->data[i];
    return s;
}

float tensor_max(const Tensor *t) {
    float m = t->data[0];
    for (size_t i = 1; i < t->size; i++)
        if (t->data[i] > m) m = t->data[i];
    return m;
}

/* -------------------------------------------------------------------------- */
/* Debug print                                                                 */
/* -------------------------------------------------------------------------- */

void tensor_print(const Tensor *t, const char *name) {
    printf("Tensor '%s' shape=[", name ? name : "?");
    for (int i = 0; i < t->ndim; i++)
        printf("%d%s", t->shape[i], i < t->ndim-1 ? "," : "");
    printf("] first 8: ");
    int lim = t->size < 8 ? (int)t->size : 8;
    for (int i = 0; i < lim; i++) printf("%.4f ", t->data[i]);
    printf("\n");
}
