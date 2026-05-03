#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/* --------------------------------------------------------------------------
 * Tensor — a flat float array with shape metadata
 * -------------------------------------------------------------------------- */

#define TENSOR_MAX_DIMS 4

typedef struct {
    float  *data;
    int     shape[TENSOR_MAX_DIMS];
    int     ndim;
    size_t  size;   /* total number of elements */
} Tensor;

/* Lifecycle */
Tensor *tensor_alloc(int ndim, int *shape);
void    tensor_free(Tensor *t);
Tensor *tensor_zeros(int ndim, int *shape);
Tensor *tensor_ones(int ndim, int *shape);
Tensor *tensor_randn(int ndim, int *shape, float std);

/* Element access helpers */
static inline float *tensor_at(Tensor *t, int i) { return t->data + i; }
static inline size_t tensor_numel(const Tensor *t) { return t->size; }

/* Basic math */
void tensor_fill(Tensor *t, float val);
void tensor_scale(Tensor *t, float s);
void tensor_add_inplace(Tensor *dst, const Tensor *src);    /* dst += src */
void tensor_mul_inplace(Tensor *dst, const Tensor *src);    /* dst *= src (elementwise) */
void tensor_copy(Tensor *dst, const Tensor *src);

/* Neural-net ops */
void tensor_matmul(Tensor *out, const Tensor *a, const Tensor *b);  /* out = a @ b */
void tensor_softmax(Tensor *t, int axis);
void tensor_layer_norm(Tensor *out, const Tensor *in,
                       const Tensor *w, const Tensor *b, float eps);
void tensor_silu(Tensor *t);     /* SiLU(x) = x * sigmoid(x) */
void tensor_tanh(Tensor *t);
void tensor_sigmoid(Tensor *t);
void tensor_relu(Tensor *t);

/* Reduction */
float tensor_sum(const Tensor *t);
float tensor_max(const Tensor *t);

/* Print */
void tensor_print(const Tensor *t, const char *name);

#endif /* TENSOR_H */
