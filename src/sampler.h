#ifndef SAMPLER_H
#define SAMPLER_H

#include "tensor.h"

/* --------------------------------------------------------------------------
 * Sampler — convert logits to a token id
 * -------------------------------------------------------------------------- */

typedef struct {
    float temperature;  /* 1.0 = unscaled; <1 = sharper; >1 = flatter */
    int   top_k;        /* 0 = disabled */
    float top_p;        /* 1.0 = disabled (nucleus sampling)           */
    unsigned int seed;
} Sampler;

void sampler_init(Sampler *s, float temperature, int top_k, float top_p,
                  unsigned int seed);

/* Sample one token index from logits */
int sampler_sample(Sampler *s, Tensor *logits);

/* Greedy argmax */
int sampler_argmax(const Tensor *logits);

#endif /* SAMPLER_H */
