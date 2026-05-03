/*
 * src/train.c — LFM training loop
 *
 * Reads a binary token file produced by scripts/prepare_data.py,
 * runs forward + manual backprop through the Liquid ODE cell,
 * updates weights with Adam, and saves the final model.
 *
 * Binary token file format (written by prepare_data.py):
 *   [magic:      uint32 = 0x4C464D44]
 *   [version:    uint32 = 1]
 *   [vocab_size: uint32]
 *   [n_tokens:   uint32]
 *   [tokens:     uint32 * n_tokens]
 */

#include "train.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>

/* ── Token file ─────────────────────────────────────────────────────────── */
#define LFMD_MAGIC 0x4C464D44u

typedef struct {
    int    *data;
    int     n;
    int     vocab_size;
} TokenFile;

static TokenFile token_file_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }

    uint32_t magic, version, vocab_size, n_tokens;
    if (fread(&magic,      4, 1, f) != 1 ||
        fread(&version,    4, 1, f) != 1 ||
        fread(&vocab_size, 4, 1, f) != 1 ||
        fread(&n_tokens,   4, 1, f) != 1) {
        fprintf(stderr, "token_file_load: bad header in %s\n", path);
        exit(1);
    }
    if (magic != LFMD_MAGIC) {
        fprintf(stderr, "token_file_load: bad magic 0x%08X (expected 0x%08X)\n",
                magic, LFMD_MAGIC);
        fprintf(stderr, "  Did you run:  python3 scripts/prepare_data.py <input.txt> %s ?\n", path);
        exit(1);
    }

    int *data = malloc(n_tokens * sizeof(int));
    if (!data) { perror("malloc"); exit(1); }
    for (uint32_t i = 0; i < n_tokens; i++) {
        uint32_t t;
        if (fread(&t, 4, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
        data[i] = (int)t;
    }
    fclose(f);
    printf("  tokens: %u  vocab: %u\n", n_tokens, vocab_size);
    return (TokenFile){ data, (int)n_tokens, (int)vocab_size };
}

/* Split flat token array into individual sequences (BOS…EOS) */
typedef struct { int *toks; int len; } Seq;

static Seq *split_seqs(TokenFile *tf, int *out_n) {
    /* count sequences: a seq starts at each BOS token (value 0)
       and ends at the next BOS or end of data */
    int cap = 1 << 16, n = 0;
    Seq *seqs = malloc(cap * sizeof(Seq));
    int i = 0;
    while (i < tf->n) {
        if (tf->data[i] != 0) { i++; continue; }
        int start = i++;
        while (i < tf->n && tf->data[i] != 0) i++;
        /* include the trailing BOS (EOS) if present */
        int end = (i < tf->n) ? i + 1 : i;
        int len = end - start;
        if (len < 3) continue;   /* too short */
        if (n >= cap) { cap *= 2; seqs = realloc(seqs, cap * sizeof(Seq)); }
        seqs[n].toks = tf->data + start;
        seqs[n].len  = len;
        n++;
    }
    *out_n = n;
    return seqs;
}

/* ── Adam state ─────────────────────────────────────────────────────────── */
typedef struct {
    float *m;     /* first moment  */
    float *v;     /* second moment */
    int    n;     /* param count   */
    int    step;
} Adam;

static Adam *adam_alloc(int n) {
    Adam *a = calloc(1, sizeof(Adam));
    a->m = calloc(n, sizeof(float));
    a->v = calloc(n, sizeof(float));
    a->n = n;
    return a;
}

static void adam_free(Adam *a) { free(a->m); free(a->v); free(a); }

static void adam_step(Adam *a, float *params, float *grad,
                      float lr, float b1, float b2, float eps, float clip) {
    /* gradient clipping */
    float norm = 0.f;
    for (int i = 0; i < a->n; i++) norm += grad[i] * grad[i];
    norm = sqrtf(norm);
    if (norm > clip) {
        float sc = clip / norm;
        for (int i = 0; i < a->n; i++) grad[i] *= sc;
    }

    a->step++;
    float lr_t = lr * sqrtf(1.f - powf(b2, a->step)) / (1.f - powf(b1, a->step));
    for (int i = 0; i < a->n; i++) {
        a->m[i] = b1 * a->m[i] + (1.f - b1) * grad[i];
        a->v[i] = b2 * a->v[i] + (1.f - b2) * grad[i] * grad[i];
        params[i] -= lr_t * a->m[i] / (sqrtf(a->v[i]) + eps);
    }
}

/* ── Flat parameter view of LFMModel (single layer) ───────────────────── */
/*
 * For simplicity this training loop handles 1-layer models only.
 * The flat param / grad arrays mirror model weights so we can pass them
 * to Adam without restructuring the model.
 */
typedef struct {
    float **ptrs;   /* pointer to each param tensor's .data */
    int    *sizes;  /* number of floats in each tensor      */
    int     n_tensors;
    int     n_total;
} FlatView;

static FlatView flat_view_alloc(LFMModel *m) {
    /* One layer only for training */
    assert(m->config.num_layers == 1);
    LFMLayerWeights *lw = &m->layers[0];
    int H = m->config.hidden_size;
    int V = m->config.vocab_size;

    /* We pack: emb, W_in, W_hh, b_h (reuse b_tau), log_tau (reuse W_tau[0..H]),
       W_out, b_out  */
    /* Actually: just use the model tensors directly — Adam needs a flat array.
       We'll copy in/out around each step. */

    /* tensors in order */
    Tensor *ts[] = {
        m->token_emb,          /* [V, H]  */
        lw->W_in,              /* [H, H]  */
        lw->W_hh,              /* [H, H]  */
        lw->b_tau,             /* [H]  — used as b_h in forward */
        lw->W_tau,             /* [H, H] — first H floats = log_tau */
        m->lm_head,            /* [V, H]  */
        m->final_ln_b,         /* [V]  — used as b_out */
    };
    int n = sizeof(ts) / sizeof(ts[0]);

    FlatView fv;
    fv.n_tensors = n;
    fv.ptrs  = malloc(n * sizeof(float *));
    fv.sizes = malloc(n * sizeof(int));
    fv.n_total = 0;
    for (int i = 0; i < n; i++) {
        fv.ptrs[i]  = ts[i]->data;
        fv.sizes[i] = (int)ts[i]->size;
        fv.n_total += fv.sizes[i];
    }
    return fv;
}

static void flat_view_free(FlatView *fv) {
    free(fv->ptrs); free(fv->sizes);
}

/* Copy model params → flat array */
static void flat_pack(FlatView *fv, float *out) {
    int off = 0;
    for (int t = 0; t < fv->n_tensors; t++) {
        memcpy(out + off, fv->ptrs[t], fv->sizes[t] * sizeof(float));
        off += fv->sizes[t];
    }
}

/* Copy flat array → model params */
static void flat_unpack(FlatView *fv, const float *in) {
    int off = 0;
    for (int t = 0; t < fv->n_tensors; t++) {
        memcpy(fv->ptrs[t], in + off, fv->sizes[t] * sizeof(float));
        off += fv->sizes[t];
    }
}

/* ── Forward + backward for one sequence ───────────────────────────────── */
/*
 * Model tensors used (single-layer, simplified LFM):
 *   emb       [V, H]   — token embedding
 *   W_in      [H, H]   — input projection
 *   W_hh      [H, H]   — recurrent weights
 *   b_tau     [H]      — hidden bias  (b_h in the ODE)
 *   W_tau     [H, H]   — first H floats = log_tau (per-neuron timescale)
 *   lm_head   [V, H]   — output projection
 *   final_ln_b[V]      — output bias
 *
 * ODE step:
 *   z[i]   = W_in[i,·]·x  +  W_hh[i,·]·h  +  b_h[i]
 *   f[i]   = tanh(z[i])
 *   τ[i]   = softplus(log_tau[i])
 *   h[i]  += dt · (-h[i] + f[i]) / τ[i]
 */

static inline float sp(float x) { return x > 20.f ? x : log1pf(expf(x)); }
static inline float sp_grad(float x) { return 1.f / (1.f + expf(-x)); } /* d/dx softplus = sigmoid */

static float fwd_bwd_seq(LFMModel *m, float *grad_flat, FlatView *fv,
                          const int *tok, int T) {
    if (T < 2) return 0.f;
    int L  = T - 1;
    int H  = m->config.hidden_size;
    int V  = m->config.vocab_size;
    int S  = m->config.ode_steps;
    float dt = 1.f / S;

    /* Param pointers */
    float *We   = m->token_emb->data;
    float *Win  = m->layers[0].W_in->data;
    float *Whh  = m->layers[0].W_hh->data;
    float *bh   = m->layers[0].b_tau->data;     /* b_h */
    float *ltau = m->layers[0].W_tau->data;     /* log_tau (first H floats) */
    float *Wo   = m->lm_head->data;
    float *bo   = m->final_ln_b->data;

    /* Grad pointers (same layout via flat_view offsets) */
    int off = 0;
    float *gWe   = grad_flat + off; off += V * H;
    float *gWin  = grad_flat + off; off += H * H;
    float *gWhh  = grad_flat + off; off += H * H;
    float *gbh   = grad_flat + off; off += H;
    float *gltau = grad_flat + off; off += H * H;  /* only first H used */
    float *gWo   = grad_flat + off; off += V * H;
    float *gbo   = grad_flat + off; /* += V */

    /* Per-step cache */
    typedef struct {
        float x[1];        /* pointer sentinel — we store tok id */
        float hin[1];      /* same */
    } _dummy;
    (void)sizeof(_dummy);

    /* allocate caches */
    float (*hpre_s)[1]  = NULL;  /* avoid VLA; use heap */
    /* We'll use a simpler flat layout */
    float *z_c    = malloc((size_t)L * S * H * sizeof(float));  /* z cache    */
    float *f_c    = malloc((size_t)L * S * H * sizeof(float));  /* f cache    */
    float *hp_c   = malloc((size_t)L * S * H * sizeof(float));  /* h_prev cache */
    float *h_c    = malloc((size_t)L * S * H * sizeof(float));  /* h_after cache */
    float *logits = malloc((size_t)L * V     * sizeof(float));
    int   *tgt    = malloc((size_t)L         * sizeof(int));
    (void)hpre_s;

#define ZC(t,s,i)  z_c   [(t)*S*H + (s)*H + (i)]
#define FC(t,s,i)  f_c   [(t)*S*H + (s)*H + (i)]
#define HPC(t,s,i) hp_c  [(t)*S*H + (s)*H + (i)]
#define HC(t,s,i)  h_c   [(t)*S*H + (s)*H + (i)]
#define LOG(t,v)   logits[(t)*V   + (v)]

    float h[4096]; /* max H=4096 */
    assert(H <= 4096);
    memset(h, 0, H * sizeof(float));
    float loss = 0.f;

    /* ── Forward ── */
    for (int t = 0; t < L; t++) {
        tgt[t] = tok[t + 1];
        float *x = We + tok[t] * H;

        for (int s = 0; s < S; s++) {
            for (int i = 0; i < H; i++) {
                HPC(t,s,i) = h[i];
                float z = bh[i];
                for (int j = 0; j < H; j++)
                    z += Win[i*H+j]*x[j] + Whh[i*H+j]*h[j];
                float f   = tanhf(z);
                float tau = sp(ltau[i]);
                ZC(t,s,i) = z;
                FC(t,s,i) = f;
                h[i] += dt * (-h[i] + f) / tau;
                HC(t,s,i) = h[i];
            }
        }

        float mx = 0.f;
        float *W_row0 = Wo;
        { float a = bo[0]; for(int j=0;j<H;j++) a+=W_row0[j]*h[j]; LOG(t,0)=a; mx=a; }
        for (int v = 1; v < V; v++) {
            float a = bo[v];
            float *row = Wo + v*H;
            for (int j = 0; j < H; j++) a += row[j] * h[j];
            LOG(t,v) = a;
            if (a > mx) mx = a;
        }
        float sum = 0.f;
        for (int v = 0; v < V; v++) { LOG(t,v) = expf(LOG(t,v)-mx); sum += LOG(t,v); }
        for (int v = 0; v < V; v++) LOG(t,v) /= sum;
        loss -= logf(LOG(t, tgt[t]) + 1e-12f);
    }

    /* ── Backward ── */
    float dh[4096];
    memset(dh, 0, H * sizeof(float));

    for (int t = L-1; t >= 0; t--) {
        float *hl = h_c + t*S*H + (S-1)*H;  /* h after last ODE step */
        float *x  = We + tok[t] * H;

        /* dlogits = probs - one_hot; grad Wo, bo */
        float dl[128]; assert(V <= 128);
        for (int v = 0; v < V; v++) dl[v] = LOG(t,v);
        dl[tgt[t]] -= 1.f;
        for (int v = 0; v < V; v++) {
            gbo[v] += dl[v];
            float *row = gWo + v*H;
            for (int j = 0; j < H; j++) {
                row[j] += dl[v] * hl[j];
                dh[j]  += Wo[v*H+j] * dl[v];
            }
        }

        /* ODE backprop */
        for (int s = S-1; s >= 0; s--) {
            float *hp = hp_c + t*S*H + s*H;
            float tau[4096], dz[4096], dztau[4096];

            for (int i = 0; i < H; i++) {
                tau[i]    = sp(ltau[i]);
                float dh_dt_tau = dh[i] * dt / tau[i];
                dz[i]     = dh_dt_tau * (1.f - FC(t,s,i)*FC(t,s,i));
                dztau[i]  = dh[i] * dt * (hp[i] - FC(t,s,i)) / (tau[i]*tau[i]);
                gltau[i] += dztau[i] * sp_grad(ltau[i]);
                gbh[i]   += dz[i];
                dh[i]    *= (1.f - dt / tau[i]);
            }

            for (int j = 0; j < H; j++) {
                float ahh=0.f, ax=0.f;
                for (int i = 0; i < H; i++) {
                    gWin[i*H+j]  += dz[i]    * x[j];
                    gWhh[i*H+j]  += dz[i]    * hp[j];
                    ahh          += Whh[i*H+j]* dz[i];
                    ax           += Win[i*H+j]* dz[i];
                }
                dh[j]              += ahh;
                gWe[tok[t]*H + j]  += ax;
            }
        }
    }

    free(z_c); free(f_c); free(hp_c); free(h_c); free(logits); free(tgt);
#undef ZC
#undef FC
#undef HPC
#undef HC
#undef LOG
    return loss / L;
}

/* ── Sampling (for progress display) ───────────────────────────────────── */
static void sample_one(LFMModel *m, float temp) {
    int H = m->config.hidden_size;
    int V = m->config.vocab_size;
    int S = m->config.ode_steps;
    float dt = 1.f / S;

    float *We   = m->token_emb->data;
    float *Win  = m->layers[0].W_in->data;
    float *Whh  = m->layers[0].W_hh->data;
    float *bh   = m->layers[0].b_tau->data;
    float *ltau = m->layers[0].W_tau->data;
    float *Wo   = m->lm_head->data;
    float *bo   = m->final_ln_b->data;

    float *h = calloc(H, sizeof(float));
    int tok = 0;   /* BOS */

    for (int step = 0; step < 30; step++) {
        float *x = We + tok * H;
        for (int s = 0; s < S; s++) {
            for (int i = 0; i < H; i++) {
                float z = bh[i];
                for (int j = 0; j < H; j++)
                    z += Win[i*H+j]*x[j] + Whh[i*H+j]*h[j];
                float tau = sp(ltau[i]);
                h[i] += dt * (-h[i] + tanhf(z)) / tau;
            }
        }
        float logits[256]; assert(V <= 256);
        float mx = -1e30f;
        for (int v = 0; v < V; v++) {
            float a = bo[v];
            for (int j = 0; j < H; j++) a += Wo[v*H+j]*h[j];
            logits[v] = a / temp;
            if (logits[v] > mx) mx = logits[v];
        }
        float sum = 0.f, pr[256];
        for (int v = 0; v < V; v++) { pr[v] = expf(logits[v]-mx); sum += pr[v]; }
        for (int v = 0; v < V; v++) pr[v] /= sum;

        float r = (float)rand() / ((float)RAND_MAX + 1.f), cum = 0.f;
        tok = V - 1;
        for (int v = 0; v < V; v++) { cum += pr[v]; if (r < cum) { tok = v; break; } }
        if (tok == 0) break;

        /* Decode: token 0 = '\n'/EOS, 1-26 = 'a'-'z' (names dataset convention) */
        if (tok >= 1 && tok <= 26) putchar('a' + tok - 1);
        else printf("[%d]", tok);
    }
    putchar('\n');
    free(h);
}

/* ── Train config defaults ──────────────────────────────────────────────── */
void train_config_defaults(TrainConfig *tc) {
    memset(tc, 0, sizeof *tc);
    tc->vocab_size        = 27;
    tc->hidden_size       = 32;
    tc->intermediate_size = 128;
    tc->num_layers        = 1;
    tc->max_seq_len       = 512;
    tc->ode_steps         = 2;
    tc->lr                = 3e-3f;
    tc->beta1             = 0.9f;
    tc->beta2             = 0.999f;
    tc->eps               = 1e-8f;
    tc->grad_clip         = 5.0f;
    tc->max_epochs        = 30;
    tc->print_every       = 1000;
    tc->sample_every      = 5000;
    tc->sample_count      = 8;
    tc->sample_temp       = 0.8f;
    tc->out_path          = "weights/model.bin";
}

/* ── Main training entry point ──────────────────────────────────────────── */
LFMModel *lfm_train(TrainConfig *tc, LFMModel *model) {
    srand((unsigned)time(NULL));

    /* Load data */
    printf("Loading data: %s\n", tc->data_path);
    TokenFile tf = token_file_load(tc->data_path);

    /* Override vocab_size from file */
    int V = tf.vocab_size;
    int H = tc->hidden_size;

    /* Allocate model if needed */
    if (!model) {
        LFMConfig cfg = {
            .vocab_size        = V,
            .hidden_size       = H,
            .intermediate_size = tc->intermediate_size,
            .num_layers        = 1,   /* training supports 1 layer */
            .max_seq_len       = tc->max_seq_len,
            .ode_steps         = tc->ode_steps,
        };
        model = lfm_model_alloc(&cfg);
        lfm_model_init_random(model);
    }

    /* We repurpose some tensors for our simplified training ODE:
       b_tau  → b_h (hidden bias, shape [H])
       W_tau  → log_tau (first H floats used as per-neuron log timescale)
       lm_head→ W_out [V, H]
       final_ln_b → b_out [V]
       Initialise log_tau = 0 → tau = softplus(0) ≈ 0.69 */
    memset(model->layers[0].W_tau->data,  0, H * sizeof(float));
    memset(model->layers[0].b_tau->data,  0, H * sizeof(float));
    tensor_fill(model->final_ln_w, 1.0f);  /* not used in training */
    tensor_fill(model->final_ln_b, 0.0f);  /* repurposed as b_out  */

    /* Build flat view for Adam */
    FlatView fv = flat_view_alloc(model);
    Adam    *adam = adam_alloc(fv.n_total);

    float *params = malloc(fv.n_total * sizeof(float));
    float *grad   = malloc(fv.n_total * sizeof(float));

    /* Split sequences */
    int   n_seqs;
    Seq  *seqs  = split_seqs(&tf, &n_seqs);
    int  *order = malloc(n_seqs * sizeof(int));
    for (int i = 0; i < n_seqs; i++) order[i] = i;

    printf("Sequences: %d\n", n_seqs);
    printf("Params: %d (%.1f KB)\n", fv.n_total, fv.n_total * 4.f / 1024.f);
    printf("Epochs: %d  LR: %.1e  H: %d  ODE_steps: %d\n\n",
           tc->max_epochs, (double)tc->lr, H, tc->ode_steps);

    long long batch = 0;
    double smooth = log((double)V);

    for (int ep = 0; ep < tc->max_epochs; ep++) {
        /* Shuffle */
        for (int i = n_seqs-1; i > 0; i--) {
            int j = rand() % (i+1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        for (int si = 0; si < n_seqs; si++, batch++) {
            Seq *s = &seqs[order[si]];

            flat_pack(&fv, params);
            memset(grad, 0, fv.n_total * sizeof(float));
            /* Sync model weights from params (first iteration is identity) */
            flat_unpack(&fv, params);

            float loss = fwd_bwd_seq(model, grad, &fv, s->toks, s->len);

            adam_step(adam, params, grad,
                      tc->lr, tc->beta1, tc->beta2, tc->eps, tc->grad_clip);
            flat_unpack(&fv, params);

            smooth = 0.999 * smooth + 0.001 * loss;

            if (batch % tc->print_every == 0)
                printf("ep %2d  batch %7lld  loss %.4f\n",
                       ep+1, batch, smooth);

            if (batch > 0 && batch % tc->sample_every == 0) {
                printf("--- samples (temp=%.1f) ---\n", tc->sample_temp);
                for (int g = 0; g < tc->sample_count; g++)
                    sample_one(model, tc->sample_temp);
                printf("---------------------------\n\n");
            }
        }
    }

    /* Final samples */
    printf("\n=== Final samples (temp=%.1f) ===\n", tc->sample_temp);
    for (int g = 0; g < 20; g++) sample_one(model, tc->sample_temp);

    /* Save */
    if (tc->out_path) {
        /* Ensure weights/ dir exists */
        system("mkdir -p weights");
        if (lfm_model_save(model, tc->out_path) == 0)
            printf("\nSaved → %s\n", tc->out_path);
    }

    free(params); free(grad);
    adam_free(adam);
    flat_view_free(&fv);
    free(seqs); free(order);
    free(tf.data);

    return model;
}
