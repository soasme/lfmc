# Training a Name Generator from Scratch

This guide walks through training a tiny **Liquid Foundation Model** to generate
human-sounding names — in pure C, no Python, no frameworks.

The model learns from 32,000 real names and after ~8 minutes on a laptop CPU it
can invent new ones like `bradynn`, `zayden`, `savika`, `malik`.

This is the LFM equivalent of Karpathy's
[makemore](https://github.com/karpathy/makemore) — same dataset, same goal,
different architecture.

---

## Prerequisites

- A C99 compiler (`gcc` or `clang`)
- `libm` (standard on Linux/macOS)
- `curl` to download the dataset

No Python. No pip. No virtual environments.

---

## Step 1 — Get the dataset

```bash
mkdir -p data
curl -o data/names.txt \
  https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

It's a plain text file: one name per line, 32,033 names, all lowercase.

```
emma
olivia
ava
isabella
sophia
...
```

---

## Step 2 — Build the trainer

```bash
gcc -O2 -march=native -D_GNU_SOURCE -o train_names train_names.c -lm
```

That's it. The entire model, training loop, and sampler live in
`train_names.c` (~400 lines).

---

## Step 3 — Train

```bash
./train_names data/names.txt
```

You'll see loss and sample names every 1,000 batches:

```
Loading data/names.txt ...
Sequences: 32033
Params: 3867  (15.1 KB)

ep  1  batch       0  loss 3.2958
ep  1  batch    1000  loss 2.8209
ep  1  batch    2000  loss 2.5575
...
ep  1  batch    5000  loss 2.3375
--- samples ---
kaya
deela
kanah
radey
...
ep 10  batch  315000  loss 2.2533
--- samples ---
kenah
celen
halley
jacyn
karzan
...
=== Final samples (temp=0.8) ===
bradynn
zayden
nina
savika
mason
malik
jaylen
kamil
```

Training takes **~8 minutes** on a modern CPU (30 epochs × 32k names).

---

## What's happening inside

### Tokenisation

Each name is a sequence of characters. The vocabulary is just 27 tokens:

| Token | Meaning |
|-------|---------|
| `0`   | `\n` — used as both BOS (start) and EOS (end) |
| `1`   | `a` |
| `2`   | `b` |
| …     | … |
| `26`  | `z` |

The name `emma` becomes the token sequence `[0, 5, 13, 13, 1, 0]`.
The model sees `[0,5,13,13,1]` as input and predicts `[5,13,13,1,0]` as targets.

### The Liquid ODE cell

At each character the hidden state `h` is updated by a mini ODE:

```
z[i]   = W_in[i,·] · x  +  W_hh[i,·] · h  +  b[i]
f[i]   = tanh(z[i])
τ[i]   = softplus(log_τ[i])          ← learned per-neuron timescale
h[i]  += dt · (-h[i] + f[i]) / τ[i]  ← Euler step
```

This is repeated `ODE_STEPS=2` times per character (dt = 0.5 each step).

Key idea: `τ[i]` is a **learned timescale** for each neuron. A small τ makes
the neuron respond quickly (short memory); a large τ makes it hold information
longer. The model learns which neurons should be fast-reacting and which
should remember context.

Contrast with a vanilla RNN:
```
h = tanh(W_in·x + W_hh·h + b)   ← discrete step, no timescale
```

The Liquid ODE cell is a smoother, more expressive version of the same idea.

### Output head

After processing a character, the hidden state is projected to logits:

```
logits[v] = W_out[v,·] · h + b_out[v]
```

Softmax over the 27-token vocabulary gives a probability for the next character.

### Loss

Standard cross-entropy:

```
L = -log p(target_token)
```

Averaged over all character positions in the name.

### Backprop through the ODE

The tricky part. The Euler integration is just arithmetic, so we can
differentiate it by hand. For a single step:

```
h_new = h + dt · (-h + f) / τ
```

Gradients flow back:

```
dL/dh      += dL/dh_new · (1 - dt/τ)        ← residual path
dL/df[i]    = dL/dh_new[i] · dt / τ[i]
dL/dτ[i]    = dL/dh_new[i] · dt · (h[i] - f[i]) / τ[i]²
dL/dlog_τ[i]= dL/dτ[i] · sigmoid(log_τ[i])  ← chain rule through softplus
dL/dz[i]    = dL/df[i] · (1 - tanh²(z[i]))  ← chain rule through tanh
```

Then the standard weight gradients:

```
dL/dW_in  += dz ⊗ x        (outer product)
dL/dW_hh  += dz ⊗ h_prev
dL/db     += dz
dL/dx     += W_in^T · dz    (flows back into the embedding)
dL/dh_prev+= W_hh^T · dz   (flows to the previous step)
```

### Optimizer

Adam with:
- Learning rate: `3e-3`
- β₁ = 0.9, β₂ = 0.999
- Gradient clipping at norm 5.0

---

## Model size

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

Edit the `#define` block at the top of `train_names.c`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `H` | 32 | Hidden size. Try 64 or 128 for better quality (slower). |
| `ODE_STEPS` | 2 | Euler steps per token. More = smoother dynamics, slower. |
| `MAX_EPOCHS` | 30 | Training epochs. 50–100 for lower loss. |
| `LR` | 3e-3 | Learning rate. Decrease if loss diverges. |
| `GRAD_CLIP` | 5.0 | Gradient clipping threshold. |

### Trying H=64

```bash
# Edit train_names.c: change H 32 → 64
gcc -O2 -march=native -D_GNU_SOURCE -o train_names train_names.c -lm
./train_names data/names.txt
```

With H=64 the model has ~15k parameters and typically reaches loss ~2.18
(vs ~2.22 for H=32), generating more convincingly name-like outputs.

---

## Sample outputs

After 30 epochs with H=32:

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

## Next steps

- **Larger model**: bump `H` to 128 or 256, add multiple layers in `src/model.c`
- **Word-level**: train on a vocabulary of words instead of characters
- **Your own data**: replace `data/names.txt` with any newline-separated text
- **Save weights**: extend `train_names.c` to call `lfm_model_save()` after training
- **LoRA fine-tuning**: coming soon in `src/optimizer.c`
