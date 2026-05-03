# lfmc — Liquid Foundation Models in C

A minimal, hackable implementation of **Liquid Foundation Models (LFM)** in pure C — no Python, no PyTorch, no CUDA runtime dependency.

Inspired by [microgpt-c](https://github.com/vixhal-baraiya/microgpt-c): if you can read it, you can modify it.

---

## What is an LFM?

Liquid Foundation Models are a family of sequence models built on **Liquid Neural Networks** — a continuous-time RNN formulation where the hidden state evolves according to an ODE:

```
dh/dt = -h/τ + f(x, h, t; θ)
```

Unlike transformers, LFMs have:
- **Linear scaling** in sequence length (no O(n²) attention)
- **Adaptive computation** — the network decides how much to process each token
- **Compact state** — the full context is a fixed-size hidden vector

LFMs were introduced by Liquid AI and are described in:
- [Liquid Foundation Models: Our First Series of Generative AI Models](https://www.liquid.ai/liquid-foundation-models)
- [LFM paper (2024)](https://arxiv.org/abs/2410.10631)

---

## Features

- [x] Forward pass (inference)
- [x] Tokenizer (BPE, byte-level)
- [x] Text generation (greedy, top-k, top-p sampling)
- [x] **Training from scratch** — full backprop through the ODE cell, Adam optimizer
- [x] **Works: names dataset** — trains a tiny LFM to generate human-sounding names ([guide](docs/training-names.md))
- [ ] LoRA fine-tuning (planned)
- [ ] Weight loading from safetensors / GGUF (planned)
- [ ] SIMD / AVX2 acceleration (planned)
- [ ] CUDA backend (planned)

---

## Quick demo — train a name generator

```bash
# 1. Build
make

# 2. Data
mkdir -p data weights
curl -o data/names.txt \
  https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
python3 scripts/prepare_data.py data/names.txt data/names_tokens.bin

# 3. Train (~8 min on CPU)
./lfmc train --data data/names_tokens.bin --out weights/names.bin
```

After ~30 epochs you get names like:

```
bradynn   zayden   nina   savika   mason
malik     jaylen   kamil  maimie   karzan
```

See the full walkthrough in **[docs/training-names.md](docs/training-names.md)**.

---

## Architecture

```
lfmc/
├── src/
│   ├── main.c          # CLI entry point (train / infer / finetune)
│   ├── model.c/h       # LFM model: layers, forward pass, ODE solver
│   ├── tensor.c/h      # N-D tensor: alloc, ops (matmul, softmax, …)
│   ├── tokenizer.c/h   # BPE tokenizer: encode / decode
│   ├── sampler.c/h     # Text sampling: greedy, top-k, top-p
│   ├── optimizer.c/h   # Adam / AdamW optimizer
│   ├── dataloader.c/h  # Streaming token data loader
│   └── utils.c/h       # Misc helpers (timing, logging, random)
├── train_names.c       # Self-contained names demo (start here)
├── docs/
│   └── training-names.md  # Step-by-step training guide
├── weights/            # Put pre-trained weight files here (gitignored)
├── data/               # Training data (gitignored)
├── Makefile
└── tests/
    └── test_tensor.c
```

---

## General inference CLI

```bash
make
./lfmc infer --weights weights/lfm-1b.bin --prompt "Once upon a time"
```

### Train (tiny model from scratch)

```bash
./lfmc train \
  --data   data/tokens.bin \
  --out    weights/lfm-tiny.bin \
  --layers 6 \
  --hidden 256 \
  --steps  10000
```

### Fine-tune with LoRA

```bash
./lfmc finetune \
  --weights weights/lfm-1b.bin \
  --data    data/tokens.bin \
  --lora-rank 8 \
  --steps   2000 \
  --out     weights/lfm-1b-finetuned.bin
```

---

## Building

Requirements: a C99 compiler, `make`, and optionally `libm` (usually already linked).

```bash
# Linux / macOS
make

# Cross-compile for ARM
CC=aarch64-linux-gnu-gcc make

# Enable AVX2 SIMD (x86)
make SIMD=avx2
```

---

## References

- [Liquid AI — LFM technical report](https://arxiv.org/abs/2410.10631)
- [Liquid Neural Networks (Hasani et al., 2022)](https://arxiv.org/abs/2006.04439)
- [microgpt-c](https://github.com/vixhal-baraiya/microgpt-c) — GPT-2 in pure C, the inspiration for this project
- [llama2.c](https://github.com/karpathy/llama2.c) — Andrej Karpathy's minimal LLaMA in C
- [makemore](https://github.com/karpathy/makemore) — Karpathy's name generator (names dataset source)

---

## License

MIT
