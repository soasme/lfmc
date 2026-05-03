#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

/* --------------------------------------------------------------------------
 * LFM (Liquid Foundation Model) — architecture
 *
 * Each layer is a Liquid Neural Network cell that solves:
 *
 *   dh/dt = -h / τ(x,h) + f(x, h; W)
 *
 * discretised with a simple Euler step (or RK4 for higher accuracy).
 * This is followed by a feedforward block (SwiGLU) and LayerNorm.
 * -------------------------------------------------------------------------- */

typedef struct {
    int vocab_size;
    int hidden_size;
    int intermediate_size;  /* FFN hidden dim, typically 4 * hidden_size */
    int num_layers;
    int max_seq_len;
    int ode_steps;          /* Euler steps per token (default: 4) */
} LFMConfig;

/* Weights for one LFM layer */
typedef struct {
    /* ODE kernel */
    Tensor *W_in;       /* [hidden, hidden] — input projection     */
    Tensor *W_hh;       /* [hidden, hidden] — recurrent weights    */
    Tensor *W_tau;      /* [hidden, hidden] — timescale gate        */
    Tensor *b_tau;      /* [hidden]                                 */

    /* SwiGLU FFN */
    Tensor *W_gate;     /* [intermediate, hidden] */
    Tensor *W_up;       /* [intermediate, hidden] */
    Tensor *W_down;     /* [hidden, intermediate] */

    /* LayerNorm */
    Tensor *ln_w;       /* [hidden] */
    Tensor *ln_b;       /* [hidden] */
} LFMLayerWeights;

/* Full model */
typedef struct {
    LFMConfig          config;
    Tensor            *token_emb;   /* [vocab_size, hidden_size]  */
    LFMLayerWeights   *layers;      /* [num_layers]               */
    Tensor            *final_ln_w;  /* [hidden_size]              */
    Tensor            *final_ln_b;  /* [hidden_size]              */
    Tensor            *lm_head;     /* [vocab_size, hidden_size]  */
} LFMModel;

/* Run state (holds activations + recurrent hidden state) */
typedef struct {
    Tensor **h;         /* per-layer hidden state [num_layers][hidden_size] */
    Tensor  *x;         /* current token embedding [hidden_size]            */
    Tensor  *logits;    /* output logits [vocab_size]                       */
} LFMRunState;

/* Lifecycle */
LFMModel    *lfm_model_alloc(LFMConfig *cfg);
void         lfm_model_free(LFMModel *m);
void         lfm_model_init_random(LFMModel *m);
int          lfm_model_save(LFMModel *m, const char *path);
LFMModel    *lfm_model_load(const char *path);

LFMRunState *lfm_run_state_alloc(LFMModel *m);
void         lfm_run_state_free(LFMRunState *s);
void         lfm_run_state_reset(LFMRunState *s);

/* Forward pass: embed token, run all layers, return logits */
void lfm_forward(LFMModel *m, LFMRunState *s, int token);

#endif /* MODEL_H */
