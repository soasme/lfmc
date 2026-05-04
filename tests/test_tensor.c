/* tests/test_tensor.c — basic sanity checks for tensor ops */
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "../src/tensor.h"

static void test_alloc_free() {
    int sh[2] = {4, 8};
    Tensor *t = tensor_zeros(2, sh);
    assert(t->ndim == 2);
    assert(t->shape[0] == 4);
    assert(t->shape[1] == 8);
    assert(t->size == 32);
    for (size_t i = 0; i < t->size; i++) assert(t->data[i] == 0.0f);
    tensor_free(t);
    printf("PASS: alloc/free\n");
}

static void test_matmul() {
    /* [2,3] @ [3,2] = [2,2] */
    int sha[2] = {2, 3};
    int shb[2] = {3, 2};
    int shc[2] = {2, 2};
    Tensor *a = tensor_zeros(2, sha);
    Tensor *b = tensor_zeros(2, shb);
    Tensor *c = tensor_zeros(2, shc);

    /* a = [[1,2,3],[4,5,6]], b = identity-ish */
    float ad[] = {1,2,3, 4,5,6};
    float bd[] = {1,0, 0,1, 0,0};
    for (int i = 0; i < 6; i++) a->data[i] = ad[i];
    for (int i = 0; i < 6; i++) b->data[i] = bd[i];

    tensor_matmul(c, a, b);
    /* c should be [[1,2],[4,5]] */
    assert(fabsf(c->data[0] - 1.0f) < 1e-5f);
    assert(fabsf(c->data[1] - 2.0f) < 1e-5f);
    assert(fabsf(c->data[2] - 4.0f) < 1e-5f);
    assert(fabsf(c->data[3] - 5.0f) < 1e-5f);

    tensor_free(a); tensor_free(b); tensor_free(c);
    printf("PASS: matmul\n");
}

static void test_softmax() {
    int sh[1] = {4};
    Tensor *t = tensor_zeros(1, sh);
    t->data[0] = 1.0f; t->data[1] = 2.0f;
    t->data[2] = 3.0f; t->data[3] = 4.0f;
    tensor_softmax(t, 0);
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += t->data[i];
    assert(fabsf(sum - 1.0f) < 1e-5f);
    assert(t->data[3] > t->data[2]);
    tensor_free(t);
    printf("PASS: softmax\n");
}

static void test_layer_norm() {
    int sh[1] = {4};
    Tensor *in = tensor_zeros(1, sh);
    Tensor *w  = tensor_ones(1, sh);
    Tensor *b  = tensor_zeros(1, sh);
    Tensor *out = tensor_zeros(1, sh);
    in->data[0] = 1; in->data[1] = 2; in->data[2] = 3; in->data[3] = 4;
    tensor_layer_norm(out, in, w, b, 1e-5f);
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += out->data[i];
    assert(fabsf(sum) < 1e-4f);  /* mean should be ~0 */
    tensor_free(in); tensor_free(w); tensor_free(b); tensor_free(out);
    printf("PASS: layer_norm\n");
}

static void test_tanh() {
    /* Test tensor_tanh: values in range [-3, 3] and outside */
    int sh[1] = {13};
    Tensor *t = tensor_zeros(1, sh);
    float inputs[] = {-4.0f, -3.0f, -2.0f, -1.0f, -0.5f, 0.0f,
                       0.5f,  1.0f,  2.0f,  3.0f,  4.0f, 0.1f, -0.1f};
    for (int i = 0; i < 13; i++) t->data[i] = inputs[i];
    tensor_tanh(t);

    /* Allow up to 0.025 error vs exact tanhf.
     * The approximation x*(27+x²)/(27+9x²) has worst-case error ~2% at |x|≈2.
     * It is intentionally a fast low-precision approximation for training where
     * exact activation values are not required. */
    for (int i = 0; i < 13; i++) {
        float expected = tanhf(inputs[i]);
        float got      = t->data[i];
        float err      = fabsf(got - expected);
        if (err > 0.025f) {
            fprintf(stderr, "FAIL: tanh[%d] input=%.3f expected=%.6f got=%.6f err=%.6f\n",
                    i, inputs[i], expected, got, err);
            assert(0);
        }
    }
    /* Clamping: |result| <= 1 always */
    for (int i = 0; i < 13; i++) {
        assert(t->data[i] >= -1.0f && t->data[i] <= 1.0f);
    }
    tensor_free(t);
    printf("PASS: tanh\n");
}

static void test_silu() {
    /* Test tensor_silu: SiLU(x) = x * sigmoid(x) */
    int sh[1] = {9};
    Tensor *t = tensor_zeros(1, sh);
    float inputs[] = {-4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f};
    for (int i = 0; i < 9; i++) t->data[i] = inputs[i];
    tensor_silu(t);

    for (int i = 0; i < 9; i++) {
        float x        = inputs[i];
        float expected = x / (1.0f + expf(-x));
        float got      = t->data[i];
        float err      = fabsf(got - expected);
        if (err > 1e-5f) {
            fprintf(stderr, "FAIL: silu[%d] input=%.3f expected=%.6f got=%.6f err=%.6f\n",
                    i, x, expected, got, err);
            assert(0);
        }
    }
    /* SiLU(0) == 0 */
    assert(fabsf(t->data[4]) < 1e-6f);
    tensor_free(t);
    printf("PASS: silu\n");
}

int main() {
    test_alloc_free();
    test_matmul();
    test_softmax();
    test_layer_norm();
    test_tanh();
    test_silu();
    printf("All tensor tests passed.\n");
    return 0;
}
