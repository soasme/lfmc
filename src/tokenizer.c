#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------- */
/* Byte-level tokenizer (256 tokens: one per byte)                            */
/* -------------------------------------------------------------------------- */

void tokenizer_init_byte_level(Tokenizer *t) {
    t->vocab_size = 256;
    t->vocab      = malloc(256 * sizeof(char *));
    t->scores     = malloc(256 * sizeof(float));
    t->bos_id     = 1;
    t->eos_id     = 2;
    t->pad_id     = 0;

    for (int i = 0; i < 256; i++) {
        char buf[5];
        snprintf(buf, sizeof(buf), "%c", (unsigned char)i);
        t->vocab[i]  = strdup(buf);
        t->scores[i] = 0.0f;
    }
}

void tokenizer_free(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
}

int tokenizer_encode(Tokenizer *t, const char *text, int *ids, int max_ids) {
    (void)t;  /* byte-level: id = byte value */
    int n = 0;
    for (const unsigned char *p = (const unsigned char *)text;
         *p && n < max_ids; p++, n++)
        ids[n] = (int)*p;
    return n;
}

const char *tokenizer_decode(Tokenizer *t, int id) {
    if (id < 0 || id >= t->vocab_size) return "<?>";
    return t->vocab[id];
}

char *tokenizer_decode_seq(Tokenizer *t, const int *ids, int n) {
    /* worst case: each token is 1 byte + null */
    char *buf = malloc((size_t)n * 4 + 1);
    int pos = 0;
    for (int i = 0; i < n; i++) {
        const char *s = tokenizer_decode(t, ids[i]);
        int len = (int)strlen(s);
        memcpy(buf + pos, s, len);
        pos += len;
    }
    buf[pos] = '\0';
    return buf;
}

/* -------------------------------------------------------------------------- */
/* File-based tokenizer (placeholder — implement BPE file format here)        */
/* -------------------------------------------------------------------------- */

int tokenizer_load(Tokenizer *t, const char *path) {
    /* TODO: load vocab from .model / .json file */
    fprintf(stderr, "tokenizer_load: not yet implemented (path=%s)\n", path);
    tokenizer_init_byte_level(t);
    return 0;
}
