#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"

/* -------------------------------------------------------------------------- */
/* Usage                                                                       */
/* -------------------------------------------------------------------------- */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s infer   --weights <file> [--prompt <text>]\n"
        "             [--steps <n>] [--temp <f>] [--top-k <n>] [--top-p <f>]\n"
        "\n"
        "  %s train   --data <tokens.bin> --out <weights.bin>\n"
        "             [--layers <n>] [--hidden <n>] [--steps <n>]\n"
        "             [--lr <f>] [--batch <n>] [--seq-len <n>]\n"
        "\n"
        "  %s finetune --weights <file> --data <tokens.bin> --out <file>\n"
        "              [--steps <n>] [--lr <f>] [--lora-rank <n>]\n",
        prog, prog, prog);
    exit(1);
}

/* -------------------------------------------------------------------------- */
/* Inference                                                                   */
/* -------------------------------------------------------------------------- */

static void cmd_infer(int argc, char **argv) {
    const char *weights  = NULL;
    const char *prompt   = "Hello";
    int         steps    = 200;
    float       temp     = 0.8f;
    int         top_k    = 40;
    float       top_p    = 0.95f;

    for (int i = 0; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--weights"))  weights = argv[++i];
        else if (!strcmp(argv[i], "--prompt"))  prompt  = argv[++i];
        else if (!strcmp(argv[i], "--steps"))   steps   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp"))    temp    = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k"))   top_k   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-p"))   top_p   = atof(argv[++i]);
    }

    if (!weights) { fprintf(stderr, "error: --weights required\n"); exit(1); }

    LFMModel *model = lfm_model_load(weights);
    if (!model) exit(1);

    Tokenizer tok;
    tokenizer_init_byte_level(&tok);

    Sampler sampler;
    sampler_init(&sampler, temp, top_k, top_p, 42);

    LFMRunState *state = lfm_run_state_alloc(model);

    /* Encode prompt */
    int ids[4096];
    int n_prompt = tokenizer_encode(&tok, prompt, ids, 4096);

    /* Feed prompt tokens */
    printf("%s", prompt);
    fflush(stdout);
    for (int i = 0; i < n_prompt; i++)
        lfm_forward(model, state, ids[i]);

    /* Generate */
    int last_token = (n_prompt > 0) ? ids[n_prompt-1] : tok.bos_id;
    for (int step = 0; step < steps; step++) {
        lfm_forward(model, state, last_token);
        last_token = sampler_sample(&sampler, state->logits);
        if (last_token == tok.eos_id) break;
        const char *tok_str = tokenizer_decode(&tok, last_token);
        printf("%s", tok_str);
        fflush(stdout);
    }
    printf("\n");

    lfm_run_state_free(state);
    tokenizer_free(&tok);
    lfm_model_free(model);
}

/* -------------------------------------------------------------------------- */
/* Train (stub — full impl in future PR)                                      */
/* -------------------------------------------------------------------------- */

static void cmd_train(int argc, char **argv) {
    const char *data_path = NULL;
    const char *out_path  = "weights/lfm.bin";
    int  num_layers  = 4;
    int  hidden_size = 256;
    int  steps       = 1000;
    float lr         = 1e-3f;

    for (int i = 0; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--data"))    data_path   = argv[++i];
        else if (!strcmp(argv[i], "--out"))     out_path    = argv[++i];
        else if (!strcmp(argv[i], "--layers"))  num_layers  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden"))  hidden_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps"))   steps       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr"))      lr          = (float)atof(argv[++i]);
    }

    if (!data_path) { fprintf(stderr, "error: --data required\n"); exit(1); }

    printf("Training LFM: layers=%d hidden=%d steps=%d lr=%.1e\n",
           num_layers, hidden_size, steps, (double)lr);
    printf("Data: %s\n", data_path);

    LFMConfig cfg = {
        .vocab_size        = 256,
        .hidden_size       = hidden_size,
        .intermediate_size = hidden_size * 4,
        .num_layers        = num_layers,
        .max_seq_len       = 512,
        .ode_steps         = 4,
    };

    LFMModel *model = lfm_model_alloc(&cfg);
    lfm_model_init_random(model);

    /* TODO: implement training loop (dataloader, forward, loss, backward, Adam) */
    printf("Training loop not yet implemented — saving randomly-initialised weights.\n");

    if (lfm_model_save(model, out_path) == 0)
        printf("Saved to %s\n", out_path);

    lfm_model_free(model);
}

/* -------------------------------------------------------------------------- */
/* Fine-tune stub                                                              */
/* -------------------------------------------------------------------------- */

static void cmd_finetune(int argc, char **argv) {
    const char *weights   = NULL;
    const char *data_path = NULL;
    const char *out_path  = "weights/lfm-finetuned.bin";
    int  steps     = 500;
    int  lora_rank = 8;
    float lr       = 5e-5f;

    for (int i = 0; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--weights"))   weights   = argv[++i];
        else if (!strcmp(argv[i], "--data"))      data_path = argv[++i];
        else if (!strcmp(argv[i], "--out"))       out_path  = argv[++i];
        else if (!strcmp(argv[i], "--steps"))     steps     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lora-rank")) lora_rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr"))        lr        = (float)atof(argv[++i]);
    }

    if (!weights || !data_path) {
        fprintf(stderr, "error: --weights and --data required\n"); exit(1);
    }

    printf("Fine-tuning: weights=%s data=%s lora_rank=%d steps=%d lr=%.1e\n",
           weights, data_path, lora_rank, steps, (double)lr);
    printf("LoRA fine-tuning not yet implemented.\n");
    (void)out_path;
}

/* -------------------------------------------------------------------------- */
/* main                                                                        */
/* -------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    if (argc < 2) usage(argv[0]);

    if (!strcmp(argv[1], "infer"))
        cmd_infer(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "train"))
        cmd_train(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "finetune"))
        cmd_finetune(argc - 2, argv + 2);
    else {
        fprintf(stderr, "Unknown command: %s\n", argv[1]);
        usage(argv[0]);
    }
    return 0;
}
