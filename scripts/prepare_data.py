#!/usr/bin/env python3
"""
prepare_data.py — Convert a text file into a binary token file for lfmc train.

Usage:
  python3 scripts/prepare_data.py <input.txt> <output.bin> [--vocab <file>]

  --vocab <file>   Path to an existing vocab file to reuse (one token per line).
                   If omitted, a new vocab is built from the input and written
                   alongside the output as <output>.vocab

Input format:
  One document (name, sentence, paragraph) per line.
  Each line becomes: BOS token + char tokens + EOS token.

Output binary format:
  [magic:     uint32 = 0x4C464D44 "LFMD"]
  [version:   uint32 = 1]
  [vocab_size:uint32]
  [n_tokens:  uint32]
  [tokens:    uint32 * n_tokens]   <- each token id as a 4-byte int

Vocab file format (.vocab):
  One token string per line (line number = token id).
  Line 0 is always the special <BOS/EOS> token (\n).

Examples:
  # Names dataset (char-level, vocab=27)
  python3 scripts/prepare_data.py data/names.txt data/names_tokens.bin

  # Reuse an existing vocab
  python3 scripts/prepare_data.py data/test.txt data/test_tokens.bin \\
          --vocab data/names_tokens.bin.vocab
"""

import sys
import struct
import argparse
from pathlib import Path


MAGIC   = 0x4C464D44   # "LFMD"
VERSION = 1
BOS_EOS = "\n"         # token 0 is both BOS and EOS


def build_vocab(lines: list[str]) -> list[str]:
    """Build a sorted character vocabulary from the data."""
    chars = set()
    for line in lines:
        chars.update(line.strip())
    # token 0 = BOS/EOS (\n), then sorted printable chars
    vocab = [BOS_EOS] + sorted(c for c in chars if c != "\n")
    return vocab


def load_vocab(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        # first line is stored as literal "\n" so we restore it
        lines = f.read().split("\n")
    vocab = []
    for tok in lines:
        if tok == "\\n":
            vocab.append("\n")
        elif tok == "":
            continue   # trailing newline
        else:
            vocab.append(tok)
    return vocab


def save_vocab(vocab: list[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for tok in vocab:
            # encode the literal newline as \n so the file is readable
            f.write("\\n\n" if tok == "\n" else tok + "\n")
    print(f"Vocab ({len(vocab)} tokens) → {path}")


def encode(lines: list[str], vocab: list[str]) -> list[int]:
    c2i = {c: i for i, c in enumerate(vocab)}
    unknown = set()
    tokens = []
    n_seqs = 0
    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            continue
        seq = [0]   # BOS
        for ch in line:
            if ch in c2i:
                seq.append(c2i[ch])
            else:
                unknown.add(ch)
        seq.append(0)   # EOS
        if len(seq) >= 3:
            tokens.extend(seq)
            n_seqs += 1
    if unknown:
        print(f"  Warning: {len(unknown)} unknown characters skipped: "
              f"{sorted(unknown)[:10]}", file=sys.stderr)
    return tokens, n_seqs


def write_bin(tokens: list[int], vocab_size: int, path: str):
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, VERSION, vocab_size, len(tokens)))
        f.write(struct.pack(f"<{len(tokens)}I", *tokens))
    kb = len(tokens) * 4 / 1024
    print(f"Binary  ({len(tokens):,} tokens, {kb:.1f} KB) → {path}")


def main():
    ap = argparse.ArgumentParser(description="Prepare text data for lfmc train")
    ap.add_argument("input",  help="Input text file (one document per line)")
    ap.add_argument("output", help="Output binary token file (.bin)")
    ap.add_argument("--vocab", default=None,
                    help="Existing .vocab file to reuse (builds new one if omitted)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Input: {args.input} ({len(lines):,} lines)")

    if args.vocab:
        vocab = load_vocab(args.vocab)
        print(f"Loaded vocab ({len(vocab)} tokens) from {args.vocab}")
    else:
        vocab = build_vocab(lines)
        vocab_path = args.output + ".vocab"
        save_vocab(vocab, vocab_path)

    tokens, n_seqs = encode(lines, vocab)
    print(f"Encoded {n_seqs:,} sequences")
    write_bin(tokens, len(vocab), args.output)
    print(f"Done. vocab_size={len(vocab)}, n_tokens={len(tokens):,}")


if __name__ == "__main__":
    main()
