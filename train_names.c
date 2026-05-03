/*
 * train_names.c — Tiny LFM trained on names.txt, pure C, no dependencies
 *
 * Architecture (single-layer):
 *   Embedding: emb[tok] → x[H]
 *   Liquid ODE cell (simplified LFM):
 *     z[i]    = W_in[i,*]·x + W_hh[i,*]·h + b_h[i]
 *     f[i]    = tanh(z[i])
 *     tau[i]  = softplus(log_tau[i])            ← per-neuron learned timescale
 *     h[i]   += dt * (-h[i] + f[i]) / tau[i]   ← Euler step (ODE_STEPS times)
 *   Output:
 *     logits[v] = W_out[v,*]·h + b_out[v]
 *   Loss: cross-entropy, optimizer: Adam
 *
 * Build:  gcc -O2 -march=native -D_GNU_SOURCE -o train_names train_names.c -lm
 * Run:    ./train_names [data/names.txt]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Hyperparameters ─────────────────────────────────────────────────── */
#define VOCAB       27      /* '\n'=0, 'a'-'z'=1-26                    */
#define H           32      /* hidden size                              */
#define ODE_STEPS    2      /* Euler integration steps per token        */
#define MAX_EPOCHS  30
#define LR          3e-3f
#define BETA1       0.9f
#define BETA2       0.999f
#define EPS_ADAM    1e-8f
#define GRAD_CLIP   5.0f
#define PRINT_EVERY 1000
#define GEN_EVERY   5000
#define GEN_N        8
#define MAX_GEN     20

/* ── Utilities ───────────────────────────────────────────────────────── */
static inline float softplus(float x) {
    /* numerically stable: log(1+exp(x)) */
    return x > 20.f ? x : log1pf(expf(x));
}
static inline float dtanh(float th) { return 1.f - th * th; } /* deriv of tanh given output */

/* Box-Muller */
static float randn(void) {
    static int have = 0; static float spare;
    if (have) { have=0; return spare; }
    float u = ((float)rand()+1.f)/((float)RAND_MAX+2.f);
    float v =  (float)rand()      /((float)RAND_MAX+1.f);
    float r = sqrtf(-2.f*logf(u));
    spare = r*sinf(2.f*(float)M_PI*v); have=1;
    return r*cosf(2.f*(float)M_PI*v);
}

/* ── Token mapping ───────────────────────────────────────────────────── */
static int c2t(char c){ return c=='\n'?0:(c>='a'&&c<='z'?c-'a'+1:-1); }
static char t2c(int t){ return t==0?'\n':(char)('a'+t-1); }

/* ── Parameters (all in one flat array) ─────────────────────────────── */
#define NP_EMB   (VOCAB*H)
#define NP_WIN   (H*H)
#define NP_WHH   (H*H)
#define NP_BH    (H)
#define NP_LTAU  (H)          /* log_tau per neuron */
#define NP_WOUT  (VOCAB*H)
#define NP_BOUT  (VOCAB)
#define NP_TOTAL (NP_EMB+NP_WIN+NP_WHH+NP_BH+NP_LTAU+NP_WOUT+NP_BOUT)

#define O_EMB   0
#define O_WIN   (O_EMB  + NP_EMB)
#define O_WHH   (O_WIN  + NP_WIN)
#define O_BH    (O_WHH  + NP_WHH)
#define O_LTAU  (O_BH   + NP_BH)
#define O_WOUT  (O_LTAU + NP_LTAU)
#define O_BOUT  (O_WOUT + NP_WOUT)

typedef struct {
    float w[NP_TOTAL];
    float m[NP_TOTAL];  /* Adam 1st moment */
    float v[NP_TOTAL];  /* Adam 2nd moment */
    int   t;            /* Adam step counter */
} Model;

static void model_init(Model *M) {
    memset(M->m, 0, sizeof M->m);
    memset(M->v, 0, sizeof M->v);
    M->t = 0;
    float s = 1.f/sqrtf(H);
    for (int i=0;i<NP_EMB; i++) M->w[O_EMB +i] = randn()*0.05f;
    for (int i=0;i<NP_WIN; i++) M->w[O_WIN +i] = randn()*s;
    for (int i=0;i<NP_WHH; i++) M->w[O_WHH +i] = randn()*s;
    for (int i=0;i<NP_BH;  i++) M->w[O_BH  +i] = 0.f;
    for (int i=0;i<NP_LTAU;i++) M->w[O_LTAU+i] = 0.f;  /* tau = softplus(0) ≈ 0.69 */
    for (int i=0;i<NP_WOUT;i++) M->w[O_WOUT+i] = randn()*0.02f;
    for (int i=0;i<NP_BOUT;i++) M->w[O_BOUT+i] = 0.f;
}

/* ── Adam update ─────────────────────────────────────────────────────── */
static void adam(Model *M, float *g) {
    M->t++;
    float lr_t = LR * sqrtf(1.f - powf(BETA2,M->t)) / (1.f - powf(BETA1,M->t));
    for (int i=0;i<NP_TOTAL;i++) {
        M->m[i] = BETA1*M->m[i] + (1.f-BETA1)*g[i];
        M->v[i] = BETA2*M->v[i] + (1.f-BETA2)*g[i]*g[i];
        M->w[i] -= lr_t * M->m[i] / (sqrtf(M->v[i]) + EPS_ADAM);
    }
}

/* ── Forward + backward for one sequence ────────────────────────────── */
/*
 * tokens[0..T-1]: BOS + chars + EOS
 * Input tokens[0..T-2], targets tokens[1..T-1]
 * Returns mean cross-entropy loss, accumulates grad into g[].
 */
static float fwd_bwd(Model *M, float *g, const int *tok, int T) {
    if (T < 2) return 0.f;
    int L = T - 1;  /* number of (input, target) pairs */

    float *We  = M->w + O_EMB;
    float *Win = M->w + O_WIN;
    float *Whh = M->w + O_WHH;
    float *bh  = M->w + O_BH;
    float *lt  = M->w + O_LTAU;
    float *Wo  = M->w + O_WOUT;
    float *bo  = M->w + O_BOUT;

    /* Cache for backprop (heap to avoid stack overflow) */
    float (*h_pre )[H]             = malloc(L * sizeof *h_pre);  /* h before ODE at t */
    float (*z_cache)[ODE_STEPS][H] = malloc(L * sizeof *z_cache);/* z = W_in@x+W_hh@h+bh */
    float (*f_cache)[ODE_STEPS][H] = malloc(L * sizeof *f_cache);/* f = tanh(z) */
    float (*h_cache)[ODE_STEPS][H] = malloc(L * sizeof *h_cache);/* h after each Euler step */
    float (*tau_s  )[ODE_STEPS][H] = malloc(L * sizeof *tau_s); /* tau per step */
    float (*hpre_s )[ODE_STEPS][H] = malloc(L * sizeof *hpre_s);/* h before each step */
    float (*logits )[VOCAB]        = malloc(L * sizeof *logits);

    float h[H]; memset(h, 0, sizeof h);
    float loss = 0.f;

    /* ── Forward ── */
    for (int t=0; t<L; t++) {
        memcpy(h_pre[t], h, H*sizeof(float));
        float *x = We + tok[t]*H;

        for (int s=0; s<ODE_STEPS; s++) {
            memcpy(hpre_s[t][s], h, H*sizeof(float));
            float dt = 1.f/ODE_STEPS;
            for (int i=0; i<H; i++) {
                float z = bh[i];
                for (int j=0; j<H; j++) z += Win[i*H+j]*x[j] + Whh[i*H+j]*h[j];
                float f  = tanhf(z);
                float tau= softplus(lt[i]);
                z_cache[t][s][i] = z;
                f_cache[t][s][i] = f;
                tau_s  [t][s][i] = tau;
                h[i] += dt * (-h[i] + f) / tau;
                h_cache[t][s][i] = h[i];
            }
        }

        /* logits = Wo @ h + bo */
        for (int v=0; v<VOCAB; v++) {
            float acc = bo[v];
            for (int j=0; j<H; j++) acc += Wo[v*H+j]*h[j];
            logits[t][v] = acc;
        }

        /* softmax + loss */
        float mx = logits[t][0];
        for (int v=1;v<VOCAB;v++) if(logits[t][v]>mx) mx=logits[t][v];
        float sum=0.f;
        for (int v=0;v<VOCAB;v++){logits[t][v]=expf(logits[t][v]-mx);sum+=logits[t][v];}
        for (int v=0;v<VOCAB;v++) logits[t][v]/=sum;  /* now = probs */
        loss -= logf(logits[t][tok[t+1]] + 1e-12f);
    }

    /* ── Backward ── */
    float dh[H]; memset(dh, 0, sizeof dh);
    const float dt = 1.f/ODE_STEPS;

    for (int t=L-1; t>=0; t--) {
        float *x = We + tok[t]*H;

        /* dlogits = probs - one_hot(target) */
        float dl[VOCAB];
        memcpy(dl, logits[t], VOCAB*sizeof(float));
        dl[tok[t+1]] -= 1.f;

        /* grad Wo, bo; and dh += Wo^T @ dl */
        float *h_last = h_cache[t][ODE_STEPS-1];
        for (int v=0;v<VOCAB;v++) {
            g[O_BOUT+v] += dl[v];
            float *row = g + O_WOUT + v*H;
            for (int j=0;j<H;j++) {
                row[j] += dl[v] * h_last[j];
                dh[j]  += Wo[v*H+j] * dl[v];
            }
        }

        /* Backprop through ODE steps (reverse) */
        for (int s=ODE_STEPS-1; s>=0; s--) {
            float *hp  = hpre_s[t][s];  /* h before this step */
            float *ff  = f_cache[t][s];
            float *tau = tau_s[t][s];
            float  dz[H], dtau_acc[H];

            for (int i=0;i<H;i++) {
                /* h_new = hp + dt*(-hp+f)/tau
                   dL/dhp  += dh*(1 - dt/tau)
                   dL/df    = dh * dt/tau
                   dL/dtau  = dh * dt * (hp - f) / tau^2
                */
                float dh_dt_o_tau = dh[i] * dt / tau[i];
                dz[i]       = dh_dt_o_tau * dtanh(ff[i]);  /* dL/dz = dL/df * f'(z) */
                dtau_acc[i] = dh[i] * dt * (hp[i] - ff[i]) / (tau[i]*tau[i]);

                /* grad log_tau: dtau/d(log_tau) = sigmoid(log_tau[i]) */
                float sig = 1.f/(1.f+expf(-M->w[O_LTAU+i]));
                g[O_LTAU+i] += dtau_acc[i] * sig;

                /* grad bh */
                g[O_BH+i] += dz[i];

                /* prepare dh for next (earlier) step */
                dh[i] *= (1.f - dt/tau[i]);  /* residual path */
            }

            /* grad Win, Whh; dh += Whh^T @ dz; dx += Win^T @ dz */
            for (int j=0;j<H;j++) {
                float acc_hh=0.f, acc_x=0.f;
                for (int i=0;i<H;i++) {
                    float *row_in  = g + O_WIN  + i*H;
                    float *row_hh  = g + O_WHH  + i*H;
                    row_in[j]  += dz[i] * x[j];
                    row_hh[j]  += dz[i] * hp[j];
                    acc_hh     += Whh[i*H+j] * dz[i];
                    acc_x      += Win[i*H+j] * dz[i];
                }
                dh[j] += acc_hh;  /* W_hh^T @ dz — gradient flows back through h */
                g[O_EMB + tok[t]*H + j] += acc_x;  /* W_in^T @ dz → embedding grad */
            }
        }
    }

    free(h_pre); free(z_cache); free(f_cache); free(h_cache);
    free(tau_s); free(hpre_s); free(logits);

    return loss / L;
}

/* ── Generate one name ───────────────────────────────────────────────── */
static void generate(Model *M, float temp) {
    float *We=M->w+O_EMB, *Win=M->w+O_WIN, *Whh=M->w+O_WHH;
    float *bh=M->w+O_BH,  *lt=M->w+O_LTAU;
    float *Wo=M->w+O_WOUT,*bo=M->w+O_BOUT;
    float h[H]; memset(h,0,sizeof h);
    int tok=0;
    for (int step=0;step<MAX_GEN;step++) {
        float *x = We + tok*H;
        float dt = 1.f/ODE_STEPS;
        for (int s=0;s<ODE_STEPS;s++) {
            for (int i=0;i<H;i++){
                float z=bh[i];
                for (int j=0;j<H;j++) z+=Win[i*H+j]*x[j]+Whh[i*H+j]*h[j];
                float f=tanhf(z), tau=softplus(lt[i]);
                h[i]+=dt*(-h[i]+f)/tau;
            }
        }
        float logits[VOCAB];
        for (int v=0;v<VOCAB;v++){
            float a=bo[v]; for(int j=0;j<H;j++) a+=Wo[v*H+j]*h[j];
            logits[v]=a/temp;
        }
        float mx=logits[0]; for(int v=1;v<VOCAB;v++) if(logits[v]>mx)mx=logits[v];
        float pr[VOCAB],sum=0.f;
        for(int v=0;v<VOCAB;v++){pr[v]=expf(logits[v]-mx);sum+=pr[v];}
        for(int v=0;v<VOCAB;v++) pr[v]/=sum;
        float r=(float)rand()/((float)RAND_MAX+1.f), cum=0.f; tok=VOCAB-1;
        for(int v=0;v<VOCAB;v++){cum+=pr[v];if(r<cum){tok=v;break;}}
        if(tok==0) break;
        putchar(t2c(tok));
    }
    putchar('\n');
}

/* ── Load data ───────────────────────────────────────────────────────── */
typedef struct { int *t; int n; } Seq;

static Seq *load(const char *path, int *nseq) {
    FILE *f=fopen(path,"r"); if(!f){perror(path);exit(1);}
    /* read all sequences into a large buffer */
    int cap=1<<20, n=0;
    int *buf=malloc(cap*sizeof(int));
    /* Also record sequence start/lengths */
    int scap=1<<16, sn=0;
    int *sstart=malloc(scap*sizeof(int));
    int *slen  =malloc(scap*sizeof(int));

    char line[128];
    while (fgets(line,sizeof line,f)) {
        int len=(int)strlen(line);
        while(len>0&&(line[len-1]=='\n'||line[len-1]=='\r'))len--;
        line[len]=0;
        if(!len) continue;
        if(n+len+3>cap){cap*=2;buf=realloc(buf,cap*sizeof(int));}
        if(sn>=scap){scap*=2;sstart=realloc(sstart,scap*sizeof(int));slen=realloc(slen,scap*sizeof(int));}
        sstart[sn]=n;
        buf[n++]=0; /* BOS */
        for(int i=0;i<len;i++){int t=c2t(line[i]);if(t>0)buf[n++]=t;}
        buf[n++]=0; /* EOS */
        slen[sn]=n-sstart[sn];
        if(slen[sn]>=3) sn++;
        else n=sstart[sn]; /* discard */
    }
    fclose(f);

    Seq *seqs=malloc(sn*sizeof(Seq));
    for(int i=0;i<sn;i++){seqs[i].t=buf+sstart[i];seqs[i].n=slen[i];}
    *nseq=sn;
    /* Note: buf not freed — seqs point into it */
    free(sstart); free(slen);
    return seqs;
}

/* ── Main ────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    const char *path = argc>1 ? argv[1] : "data/names.txt";
    srand((unsigned)time(NULL));

    printf("Loading %s ...\n", path);
    int nseq;
    Seq *seqs = load(path, &nseq);
    printf("Sequences: %d\n", nseq);
    printf("Params: %d  (%.1f KB)\n\n", NP_TOTAL, NP_TOTAL*4.f/1024.f);

    Model *M = calloc(1, sizeof(Model));
    model_init(M);

    float *grad = calloc(NP_TOTAL, sizeof(float));
    int *order  = malloc(nseq * sizeof(int));
    for (int i=0;i<nseq;i++) order[i]=i;

    long long batch=0;
    double smooth = log(VOCAB); /* start at uniform entropy */

    for (int ep=0; ep<MAX_EPOCHS; ep++) {
        /* shuffle */
        for(int i=nseq-1;i>0;i--){int j=rand()%(i+1);int tmp=order[i];order[i]=order[j];order[j]=tmp;}

        for (int si=0; si<nseq; si++, batch++) {
            Seq *s = &seqs[order[si]];
            memset(grad, 0, NP_TOTAL*sizeof(float));

            float loss = fwd_bwd(M, grad, s->t, s->n);

            /* gradient clipping */
            float norm=0.f;
            for(int i=0;i<NP_TOTAL;i++) norm+=grad[i]*grad[i];
            norm=sqrtf(norm);
            if(norm>GRAD_CLIP){float sc=GRAD_CLIP/norm;for(int i=0;i<NP_TOTAL;i++)grad[i]*=sc;}

            adam(M, grad);
            smooth = 0.999*smooth + 0.001*loss;

            if(batch % PRINT_EVERY == 0)
                printf("ep %2d  batch %7lld  loss %.4f\n", ep+1, batch, smooth);

            if(batch % GEN_EVERY == 0 && batch > 0) {
                printf("--- samples ---\n");
                for(int g=0;g<GEN_N;g++) generate(M, 0.8f);
                printf("---------------\n\n");
            }
        }
    }

    printf("\n=== Final samples (temp=0.8) ===\n");
    for(int g=0;g<20;g++) generate(M, 0.8f);
    printf("\n=== Final samples (temp=1.0) ===\n");
    for(int g=0;g<20;g++) generate(M, 1.0f);

    return 0;
}
