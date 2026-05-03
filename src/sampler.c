#include "sampler.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

void sampler_init(Sampler *s, float temperature, int top_k, float top_p,
                  unsigned int seed) {
    s->temperature = temperature;
    s->top_k       = top_k;
    s->top_p       = top_p;
    s->seed        = seed;
    srand(seed);
}

int sampler_argmax(const Tensor *logits) {
    int best = 0;
    for (int i = 1; i < (int)logits->size; i++)
        if (logits->data[i] > logits->data[best]) best = i;
    return best;
}

/* Simple comparison for qsort (descending by value) */
typedef struct { float val; int idx; } IndexedFloat;
static int cmp_desc(const void *a, const void *b) {
    float da = ((IndexedFloat *)a)->val;
    float db = ((IndexedFloat *)b)->val;
    return (da < db) - (da > db);
}

int sampler_sample(Sampler *s, Tensor *logits) {
    int n = (int)logits->size;

    if (s->temperature <= 0.0f || (s->top_k == 1))
        return sampler_argmax(logits);

    /* Apply temperature */
    float *ldata = logits->data;
    float inv_t  = 1.0f / s->temperature;

    /* Build sorted index array */
    IndexedFloat *buf = malloc(n * sizeof(IndexedFloat));
    for (int i = 0; i < n; i++) { buf[i].val = ldata[i] * inv_t; buf[i].idx = i; }
    qsort(buf, n, sizeof(IndexedFloat), cmp_desc);

    /* top-k truncation */
    int k = (s->top_k > 0 && s->top_k < n) ? s->top_k : n;

    /* Softmax over top-k */
    float max_val = buf[0].val;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        buf[i].val = expf(buf[i].val - max_val);
        sum += buf[i].val;
    }
    for (int i = 0; i < k; i++) buf[i].val /= sum;

    /* top-p nucleus truncation */
    if (s->top_p < 1.0f) {
        float cumsum = 0.0f;
        int nucleus = k;
        for (int i = 0; i < k; i++) {
            cumsum += buf[i].val;
            if (cumsum >= s->top_p) { nucleus = i + 1; break; }
        }
        k = nucleus;
        /* renormalize */
        sum = 0.0f;
        for (int i = 0; i < k; i++) sum += buf[i].val;
        for (int i = 0; i < k; i++) buf[i].val /= sum;
    }

    /* Sample */
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cumsum = 0.0f;
    int chosen = buf[k-1].idx;
    for (int i = 0; i < k; i++) {
        cumsum += buf[i].val;
        if (r < cumsum) { chosen = buf[i].idx; break; }
    }

    free(buf);
    return chosen;
}
