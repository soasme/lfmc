#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

static Tensor *mat(int rows, int cols) {
    int sh[2] = {rows, cols};
    return tensor_zeros(2, sh);
}

static Tensor *vec(int n) {
    int sh[1] = {n};
    return tensor_zeros(1, sh);
}

/* -------------------------------------------------------------------------- */
/* Lifecycle                                                                   */
/* -------------------------------------------------------------------------- */

LFMModel *lfm_model_alloc(LFMConfig *cfg) {
    LFMModel *m = calloc(1, sizeof(LFMModel));
    m->config = *cfg;

    int H = cfg->hidden_size;
    int I = cfg->intermediate_size;
    int V = cfg->vocab_size;

    int sh2[2]; (void)sh2;

    m->token_emb  = mat(V, H);
    m->final_ln_w = vec(H);
    m->final_ln_b = vec(H);
    m->lm_head    = mat(V, H);

    m->layers = calloc(cfg->num_layers, sizeof(LFMLayerWeights));
    for (int l = 0; l < cfg->num_layers; l++) {
        LFMLayerWeights *lw = &m->layers[l];
        lw->W_in   = mat(H, H);
        lw->W_hh   = mat(H, H);
        lw->W_tau  = mat(H, H);
        lw->b_tau  = vec(H);
        lw->W_gate = mat(I, H);
        lw->W_up   = mat(I, H);
        lw->W_down = mat(H, I);
        lw->ln_w   = vec(H);
        lw->ln_b   = vec(H);
    }
    return m;
}

void lfm_model_free(LFMModel *m) {
    if (!m) return;
    tensor_free(m->token_emb);
    tensor_free(m->final_ln_w);
    tensor_free(m->final_ln_b);
    tensor_free(m->lm_head);
    for (int l = 0; l < m->config.num_layers; l++) {
        LFMLayerWeights *lw = &m->layers[l];
        tensor_free(lw->W_in);
        tensor_free(lw->W_hh);
        tensor_free(lw->W_tau);
        tensor_free(lw->b_tau);
        tensor_free(lw->W_gate);
        tensor_free(lw->W_up);
        tensor_free(lw->W_down);
        tensor_free(lw->ln_w);
        tensor_free(lw->ln_b);
    }
    free(m->layers);
    free(m);
}

void lfm_model_init_random(LFMModel *m) {
    int H = m->config.hidden_size;
    float std = 0.02f;

    /* Helper: fill with randn scaled by std */
#define RANDN(t, s) do { \
    Tensor *_tmp = tensor_randn((t)->ndim, (t)->shape, (s)); \
    tensor_copy((t), _tmp); \
    tensor_free(_tmp); \
} while(0)

    RANDN(m->token_emb, std);
    tensor_fill(m->final_ln_w, 1.0f);
    tensor_fill(m->final_ln_b, 0.0f);
    RANDN(m->lm_head, std);

    for (int l = 0; l < m->config.num_layers; l++) {
        LFMLayerWeights *lw = &m->layers[l];
        float s = 1.0f / sqrtf((float)H);
        RANDN(lw->W_in,  s);
        RANDN(lw->W_hh,  s);
        RANDN(lw->W_tau, s);
        tensor_fill(lw->b_tau, 1.0f);  /* init τ bias ≈ 1 */
        RANDN(lw->W_gate, std);
        RANDN(lw->W_up,   std);
        RANDN(lw->W_down, std);
        tensor_fill(lw->ln_w, 1.0f);
        tensor_fill(lw->ln_b, 0.0f);
    }
#undef RANDN
}

/* -------------------------------------------------------------------------- */
/* Run state                                                                   */
/* -------------------------------------------------------------------------- */

LFMRunState *lfm_run_state_alloc(LFMModel *m) {
    LFMRunState *s = calloc(1, sizeof(LFMRunState));
    int H = m->config.hidden_size;
    int V = m->config.vocab_size;
    s->h = malloc(m->config.num_layers * sizeof(Tensor *));
    for (int l = 0; l < m->config.num_layers; l++)
        s->h[l] = vec(H);
    s->x      = vec(H);
    s->logits = vec(V);
    return s;
}

void lfm_run_state_free(LFMRunState *s) {
    if (!s) return;
    /* Note: num_layers not stored here; caller should call reset first if needed */
    free(s->h);
    tensor_free(s->x);
    tensor_free(s->logits);
    free(s);
}

void lfm_run_state_reset(LFMRunState *s) {
    /* Zero all hidden states between independent sequences */
    /* (caller must know num_layers) */
}

/* -------------------------------------------------------------------------- */
/* Forward pass                                                                */
/* -------------------------------------------------------------------------- */

/*
 * One LFM layer forward:
 *
 *   For t = 0 .. ode_steps-1:
 *     f    = tanh(W_in @ x + W_hh @ h)
 *     tau  = softplus(W_tau @ h + b_tau)          ← learned timescale
 *     dh   = (-h + f) / tau
 *     h   += dh / ode_steps                       ← Euler step
 *
 *   Then SwiGLU FFN + residual + LayerNorm
 */
static void lfm_layer_forward(LFMLayerWeights *lw, Tensor *h, const Tensor *x,
                               int ode_steps, int H, int I) {
    Tensor *f     = vec(H);
    Tensor *tau   = vec(H);
    Tensor *tmp   = vec(H);
    Tensor *gate  = vec(I);
    Tensor *up    = vec(I);
    Tensor *ffn   = vec(H);
    Tensor *h_in  = vec(H);
    float dt = 1.0f / (float)ode_steps;

    /* ODE integration */
    for (int step = 0; step < ode_steps; step++) {
        /* f = W_in @ x + W_hh @ h */
        tensor_matmul(f,   lw->W_in, x);
        tensor_matmul(tmp, lw->W_hh, h);
        tensor_add_inplace(f, tmp);
        tensor_tanh(f);

        /* tau = softplus(W_tau @ h + b_tau) */
        tensor_matmul(tau, lw->W_tau, h);
        tensor_add_inplace(tau, lw->b_tau);
        /* softplus(x) = log(1 + exp(x)) */
        for (int i = 0; i < H; i++)
            tau->data[i] = log1pf(expf(tau->data[i])) + 1e-6f;

        /* dh = (-h + f) / tau  →  h += dh * dt */
        for (int i = 0; i < H; i++)
            h->data[i] += dt * (-h->data[i] + f->data[i]) / tau->data[i];
    }

    /* Save h before residual */
    tensor_copy(h_in, h);

    /* SwiGLU FFN: out = W_down @ (SiLU(W_gate @ h) * (W_up @ h)) */
    tensor_matmul(gate, lw->W_gate, h);
    tensor_matmul(up,   lw->W_up,   h);
    tensor_silu(gate);
    tensor_mul_inplace(gate, up);       /* gate = SiLU(gate) * up */
    tensor_matmul(ffn, lw->W_down, gate);

    /* Residual + LayerNorm */
    tensor_add_inplace(ffn, h_in);
    tensor_layer_norm(h, ffn, lw->ln_w, lw->ln_b, 1e-5f);

    tensor_free(f);
    tensor_free(tau);
    tensor_free(tmp);
    tensor_free(gate);
    tensor_free(up);
    tensor_free(ffn);
    tensor_free(h_in);
}

void lfm_forward(LFMModel *m, LFMRunState *s, int token) {
    LFMConfig *c = &m->config;
    int H = c->hidden_size;
    int I = c->intermediate_size;
    int V = c->vocab_size;

    assert(token >= 0 && token < V);

    /* Embed token: x = token_emb[token] */
    memcpy(s->x->data, m->token_emb->data + (size_t)token * H,
           H * sizeof(float));

    /* Run each layer */
    for (int l = 0; l < c->num_layers; l++)
        lfm_layer_forward(&m->layers[l], s->h[l], s->x,
                          c->ode_steps, H, I);

    /* Use last layer's hidden state as output */
    Tensor *out = vec(H);
    tensor_layer_norm(out, s->h[c->num_layers-1],
                      m->final_ln_w, m->final_ln_b, 1e-5f);

    /* lm_head: logits = lm_head @ out */
    /* lm_head is [V, H], so logits[v] = dot(lm_head[v], out) */
    for (int v = 0; v < V; v++) {
        float dot = 0.0f;
        float *row = m->lm_head->data + (size_t)v * H;
        for (int i = 0; i < H; i++) dot += row[i] * out->data[i];
        s->logits->data[v] = dot;
    }

    tensor_free(out);
}

/* -------------------------------------------------------------------------- */
/* Save / load (simple flat binary format)                                    */
/* -------------------------------------------------------------------------- */

static void write_tensor(FILE *f, Tensor *t) {
    fwrite(&t->ndim, sizeof(int), 1, f);
    fwrite(t->shape, sizeof(int), t->ndim, f);
    fwrite(t->data,  sizeof(float), t->size, f);
}

static Tensor *read_tensor(FILE *f) {
    int ndim;
    fread(&ndim, sizeof(int), 1, f);
    int shape[TENSOR_MAX_DIMS];
    fread(shape, sizeof(int), ndim, f);
    Tensor *t = tensor_alloc(ndim, shape);
    fread(t->data, sizeof(float), t->size, f);
    return t;
}

int lfm_model_save(LFMModel *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    /* Magic + version */
    uint32_t magic = 0x4C464D43; /* "LFMC" */
    uint32_t ver   = 1;
    fwrite(&magic, 4, 1, f);
    fwrite(&ver,   4, 1, f);

    fwrite(&m->config, sizeof(LFMConfig), 1, f);
    write_tensor(f, m->token_emb);
    write_tensor(f, m->final_ln_w);
    write_tensor(f, m->final_ln_b);
    write_tensor(f, m->lm_head);
    for (int l = 0; l < m->config.num_layers; l++) {
        LFMLayerWeights *lw = &m->layers[l];
        write_tensor(f, lw->W_in);
        write_tensor(f, lw->W_hh);
        write_tensor(f, lw->W_tau);
        write_tensor(f, lw->b_tau);
        write_tensor(f, lw->W_gate);
        write_tensor(f, lw->W_up);
        write_tensor(f, lw->W_down);
        write_tensor(f, lw->ln_w);
        write_tensor(f, lw->ln_b);
    }
    fclose(f);
    return 0;
}

LFMModel *lfm_model_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    uint32_t magic, ver;
    fread(&magic, 4, 1, f);
    fread(&ver,   4, 1, f);
    if (magic != 0x4C464D43) {
        fprintf(stderr, "lfm_model_load: bad magic\n");
        fclose(f); return NULL;
    }

    LFMConfig cfg;
    fread(&cfg, sizeof(LFMConfig), 1, f);
    LFMModel *m = lfm_model_alloc(&cfg);

    /* Replace zero tensors with loaded ones */
    tensor_free(m->token_emb);  m->token_emb  = read_tensor(f);
    tensor_free(m->final_ln_w); m->final_ln_w = read_tensor(f);
    tensor_free(m->final_ln_b); m->final_ln_b = read_tensor(f);
    tensor_free(m->lm_head);    m->lm_head    = read_tensor(f);

    for (int l = 0; l < cfg.num_layers; l++) {
        LFMLayerWeights *lw = &m->layers[l];
        tensor_free(lw->W_in);   lw->W_in   = read_tensor(f);
        tensor_free(lw->W_hh);   lw->W_hh   = read_tensor(f);
        tensor_free(lw->W_tau);  lw->W_tau  = read_tensor(f);
        tensor_free(lw->b_tau);  lw->b_tau  = read_tensor(f);
        tensor_free(lw->W_gate); lw->W_gate = read_tensor(f);
        tensor_free(lw->W_up);   lw->W_up   = read_tensor(f);
        tensor_free(lw->W_down); lw->W_down = read_tensor(f);
        tensor_free(lw->ln_w);   lw->ln_w   = read_tensor(f);
        tensor_free(lw->ln_b);   lw->ln_b   = read_tensor(f);
    }
    fclose(f);
    return m;
}
