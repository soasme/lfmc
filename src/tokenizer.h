#ifndef TOKENIZER_H
#define TOKENIZER_H

/* --------------------------------------------------------------------------
 * Tokenizer — byte-level BPE (compatible with GPT-2 / tiktoken cl100k style)
 *
 * For now this is a character-level (byte) tokenizer suitable for training
 * small models from scratch.  A full BPE tokenizer can be swapped in later.
 * -------------------------------------------------------------------------- */

typedef struct {
    char  **vocab;      /* id → string (owned) */
    float  *scores;     /* merge scores (BPE priority) */
    int     vocab_size;

    /* Special tokens */
    int bos_id;
    int eos_id;
    int pad_id;
} Tokenizer;

int  tokenizer_load(Tokenizer *t, const char *path);
void tokenizer_free(Tokenizer *t);

/* Byte-level fallback: each byte is a token (id = byte value, 0..255) */
void tokenizer_init_byte_level(Tokenizer *t);

/* Encode a UTF-8 string → token ids.  Returns number of tokens. */
int  tokenizer_encode(Tokenizer *t, const char *text, int *ids, int max_ids);

/* Decode a token id → string (pointer valid until next call) */
const char *tokenizer_decode(Tokenizer *t, int id);

/* Decode a sequence of ids into a malloc'd string (caller frees) */
char *tokenizer_decode_seq(Tokenizer *t, const int *ids, int n);

#endif /* TOKENIZER_H */
