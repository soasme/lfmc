# Training a Name Generator from Scratch

This guide walks through training a tiny **Liquid Foundation Model** to generate
human-sounding names — using the `lfmc train` CLI, in pure C, no Python frameworks.

The model learns from 32,000 real names and after ~8 minutes on a laptop CPU
generates new ones like `bradynn`, `zayden`, `savika`, `malik`.

This is the LFM equivalent of Karpathy's
[makemore](https://github.com/karpathy/makemore) — same dataset, same goal,
different architecture.

---

## Prerequisites

- A C99 compiler (`gcc` or `clang`)
- `make`
- Python 3 (for data preparation only — one script, no pip installs)
- `curl` to download the dataset

---

## Step 1 — Build

```bash
make
# produces ./lfmc
```

---

## Step 2 — Get the dataset

```bash
mkdir -p data
curl -o data/names.txt \
  https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

Plain text, one name per line, 32,033 names, all lowercase:

```
emma
olivia
ava
isabella
sophia
...
```

---

## Step 3 — Prepare the data

`lfmc train` expects a binary token file. Convert the text with:

```bash
python3 scripts/prepare_data.py data/names.txt data/names_tokens.bin
```

Output:

```
Input: data/names.txt (32,033 lines)
Vocab (27 tokens) → data/names_tokens.bin.vocab
Encoded 32,033 sequences
Binary  (260,179 tokens, 1016.3 KB) → data/names_tokens.bin
Done. vocab_size=27, n_tokens=260,179
```

Two files are created:
- `data/names_tokens.bin` — binary token file read by `lfmc train`
- `data/names_tokens.bin.vocab` — human-readable vocab (one token per line)

The vocab for the names dataset is 27 tokens:

| Token | Character |
|-------|-----------|
| 0     | `\n` — BOS and EOS |
| 1     | `a` |
| 2     | `b` |
| …     | … |
| 26    | `z` |

Each name is encoded as `[0, chars..., 0]` — the newline token doubles as
both the start-of-name and end-of-name signal.

---

## Step 4 — Train

```bash
mkdir -p weights
./lfmc train \
  --data   data/names_tokens.bin \
  --out    weights/names.bin \
  --hidden 32 \
  --epochs 30
```

You'll see loss and sample names printed every 1,000 batches:

```
Loading data: data/names_tokens.bin
  tokens: 260,179  vocab: 27
Sequences: 32033
Params: 3867 (15.1 KB)
Epochs: 30  LR: 3.0e-03  H: 32  ODE_steps: 2

ep  1  batch       0  loss 3.2958
ep  1  batch    1000  loss 2.8209
ep  1  batch    2000  loss 2.5575
...
ep  1  batch    5000  loss 2.3375
--- samples (temp=0.8) ---
kaya
deela
radey
...
ep 10  batch  315000  loss 2.2533
--- samples (temp=0.8) ---
halley
jacyn
karzan
maimie
...

=== Final samples (temp=0.8) ===
bradynn    zayden    nina      savika
mason      malik     jaylen    kamil

Saved → weights/names.bin
```

Training takes about **8 minutes** on a modern CPU (30 epochs × 32k sequences).

---

## Full pipeline at a glance

```bash
# 1. Build
make

# 2. Data
mkdir -p data weights
curl -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
python3 scripts/prepare_data.py data/names.txt data/names_tokens.bin

# 3. Train
./lfmc train --data data/names_tokens.bin --out weights/names.bin

# 4. Infer  (once inference for char-level models is wired up)
./lfmc infer --weights weights/names.bin --steps 20
```

---

## What's happening inside

### The Liquid ODE cell

At each character the hidden state `h ∈ ℝᴴ` is updated by integrating an ODE:

```
z[i]   = W_in[i,·] · x  +  W_hh[i,·] · h  +  b[i]
f[i]   = tanh(z[i])
τ[i]   = softplus(log_τ[i])           ← learned per-neuron timescale
h[i]  += dt · (-h[i] + f[i]) / τ[i]  ← Euler step
```

This is repeated `ODE_STEPS=2` times per character (dt = 0.5 each).

The key ingredient is `τ[i]`: a **learned timescale** for each neuron. A small
τ makes a neuron respond quickly (short memory); a large τ makes it hold
information longer. The model learns which neurons should be fast and which
should remember context.

Contrast with a vanilla RNN:
```
h = tanh(W_in·x + W_hh·h + b)   ← discrete, no timescale
```

The ODE formulation gives a smoother, more expressive update rule.

### Backprop through the ODE

The Euler step is just arithmetic, so we differentiate it by hand.
For one step `h_new = h + dt·(-h+f)/τ`:

```
dL/dh       += dL/dh_new · (1 - dt/τ)           ← residual path
dL/df[i]     = dL/dh_new[i] · dt / τ[i]
dL/dτ[i]     = dL/dh_new[i] · dt · (h[i]-f[i]) / τ[i]²
dL/dlog_τ[i] = dL/dτ[i] · sigmoid(log_τ[i])     ← chain rule through softplus
dL/dz[i]     = dL/df[i] · (1 - tanh²(z[i]))     ← chain rule through tanh
```

Then the standard weight gradients:

```
dL/dW_in  += dz ⊗ x        (outer product, accumulated over sequence)
dL/dW_hh  += dz ⊗ h_prev
dL/db     += dz
dL/dx     += W_in^T · dz   → flows back into the embedding
dL/dh_prev+= W_hh^T · dz  → flows to the previous ODE step
```

All of this lives in `src/train.c: fwd_bwd_seq()`.

### Optimizer

Adam with:
- Learning rate: `3e-3`
- β₁ = 0.9, β₂ = 0.999
- Gradient clipping at norm 5.0

---

## Model size

Default configuration (`--hidden 32`):

| Component | Shape | Parameters |
|-----------|-------|-----------|
| Embedding | [27, 32] | 864 |
| W_in | [32, 32] | 1,024 |
| W_hh | [32, 32] | 1,024 |
| b_h | [32] | 32 |
| log_τ | [32] | 32 |
| W_out | [27, 32] | 864 |
| b_out | [27] | 27 |
| **Total** | | **3,867** |

That's a **15 KB model** — the entire network fits in L1 cache.

---

## Tuning

| Flag | Default | Effect |
|------|---------|--------|
| `--hidden <n>` | 32 | Hidden size. Try 64 or 128 for better quality (slower). |
| `--ode-steps <n>` | 2 | Euler steps per token. More = smoother dynamics, slower. |
| `--epochs <n>` | 30 | Training epochs. 50–100 for lower loss. |
| `--lr <f>` | 3e-3 | Learning rate. Decrease if loss goes NaN. |

### Trying H=64

```bash
./lfmc train \
  --data   data/names_tokens.bin \
  --out    weights/names-h64.bin \
  --hidden 64 \
  --epochs 50
```

With H=64 (~15k params) the model typically reaches loss ~2.18 vs ~2.22 for
H=32, generating more convincingly name-like outputs.

---

## Sample outputs (H=32, 30 epochs)

**temp=0.8** (conservative):
```
bradynn    zayden     nina      savika
mason      malik      jaylen    kamil
maimie     karzan     halley    jacyn
```

**temp=1.0** (more creative):
```
zayden     surais     nyeria    claleel
alyah      balelah    byllen    bridyn
zahya      jacen      selokin   adran
```

---

## Training on your own data

`prepare_data.py` works on any newline-delimited text:

```bash
# One sentence per line
python3 scripts/prepare_data.py data/sentences.txt data/sentences_tokens.bin

# Reuse an existing vocab (so token ids stay consistent)
python3 scripts/prepare_data.py data/test.txt data/test_tokens.bin \
        --vocab data/sentences_tokens.bin.vocab

# Then train
./lfmc train --data data/sentences_tokens.bin --hidden 128 --epochs 20
```

---

## Next steps

- **More layers**: multi-layer support is in `src/model.h` — extend `lfm_train` in `src/train.c`
- **Save + reload**: trained weights are saved as `.bin` and can be reloaded with `lfm_model_load()`
- **LoRA fine-tuning**: coming soon (`./lfmc finetune`)
- **Download pre-trained weights**: `bash scripts/download-model.sh`
