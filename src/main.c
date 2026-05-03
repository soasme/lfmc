#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "train.h"
#include "tokenizer.h"
#include "sampler.h"

/* -------------------------------------------------------------------------- */
/* Usage                                                                       */
/* -------------------------------------------------------------------------- */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s train   --data <tokens.bin> [options]\n"
        "\n"
        "    --data   <file>   Token file from scripts/prepare_data.py  (required)\n"
        "    --out    <file>   Where to save the model  [weights/model.bin]\n"
        "    --hidden <n>      Hidden size              [32]\n"
        "    --layers <n>      Number of layers         [1]\n"
        "    --ode-steps <n>   Euler steps per token    [2]\n"
        "    --epochs <n>      Training epochs          [30]\n"
        "    --lr     <f>      Learning rate            [3e-3]\n"
        "\n"
        "  %s infer   --weights <file> [options]\n"
        "\n"
        "    --weights <file>  Trained weight file      (required)\n"
        "    --prompt  <text>  Prompt text              [\"\"]\n"
        "    --steps   <n>     Tokens to generate       [200]\n"
        "    --temp    <f>     Sampling temperature     [0.8]\n"
        "    --top-k   <n>     Top-k sampling           [40]\n"
        "    --top-p   <f>     Top-p (nucleus) sampling [0.95]\n"
        "\n"
        "  %s finetune --weights <file> --data <tokens.bin> [options]\n"
        "\n"
        "    --weights <file>  Base model to fine-tune  (required)\n"
        "    --data    <file>  Token file               (required)\n"
        "    --out     <file>  Output path              [weights/finetuned.bin]\n"
        "    --lora-rank <n>   LoRA rank                [8]\n"
        "    --epochs  <n>     Fine-tuning epochs       [5]\n"
        "    --lr      <f>     Learning rate            [5e-5]\n"
        "\n"
        "Data preparation:\n"
        "  python3 scripts/prepare_data.py <input.txt> <output.bin>\n"
        "\n"
        "Download pre-trained weights:\n"
        "  bash scripts/download-model.sh\n"
        "\n",
        prog, prog, prog);
    exit(1);
}

/* -------------------------------------------------------------------------- */
/* Train                                                                       */
/* -------------------------------------------------------------------------- */

static void cmd_train(int argc, char **argv) {
    TrainConfig tc;
    train_config_defaults(&tc);

    for (int i = 0; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--data"))      tc.data_path          = argv[++i];
        else if (!strcmp(argv[i], "--out"))        tc.out_path           = argv[++i];
        else if (!strcmp(argv[i], "--hidden"))     tc.hidden_size        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers"))     tc.num_layers         = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ode-steps"))  tc.ode_steps          = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--epochs"))     tc.max_epochs         = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr"))         tc.lr                 = (float)atof(argv[++i]);
    }
    /* last arg might be data path without flag */
    if (!tc.data_path && argc > 0) tc.data_path = argv[argc - 1];

    if (!tc.data_path) {
        fprintf(stderr, "error: --data <tokens.bin> is required\n");
        fprintf(stderr, "  Prepare with: python3 scripts/prepare_data.py names.txt data/names_tokens.bin\n");
        exit(1);
    }

    LFMModel *model = lfm_train(&tc, NULL);
    lfm_model_free(model);
}

/* -------------------------------------------------------------------------- */
/* Inference                                                                   */
/* -------------------------------------------------------------------------- */

static void cmd_infer(int argc, char **argv) {
    const char *weights  = NULL;
    const char *prompt   = "";
    int         steps    = 200;
    float       temp     = 0.8f;
    int         top_k    = 40;
    float       top_p    = 0.95f;

    for (int i = 0; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--weights"))  weights = argv[++i];
        else if (!strcmp(argv[i], "--prompt"))   prompt  = argv[++i];
        else if (!strcmp(argv[i], "--steps"))    steps   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp"))     temp    = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k"))    top_k   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-p"))    top_p   = (float)atof(argv[++i]);
    }

    if (!weights) { fprintf(stderr, "error: --weights required\n"); exit(1); }

    LFMModel *model = lfm_model_load(weights);
    if (!model) exit(1);

    Tokenizer tok;
    tokenizer_init_byte_level(&tok);

    Sampler sampler;
    sampler_init(&sampler, temp, top_k, top_p, 42);

    LFMRunState *state = lfm_run_state_alloc(model);

    int ids[4096];
    int n_prompt = tokenizer_encode(&tok, prompt, ids, 4096);

    if (n_prompt > 0) printf("%s", prompt);
    fflush(stdout);
    for (int i = 0; i < n_prompt; i++)
        lfm_forward(model, state, ids[i]);

    int last_token = (n_prompt > 0) ? ids[n_prompt - 1] : tok.bos_id;
    for (int step = 0; step < steps; step++) {
        lfm_forward(model, state, last_token);
        last_token = sampler_sample(&sampler, state->logits);
        if (last_token == tok.eos_id) break;
        printf("%s", tokenizer_decode(&tok, last_token));
        fflush(stdout);
    }
    printf("\n");

    lfm_run_state_free(state);
    tokenizer_free(&tok);
    lfm_model_free(model);
}

/* -------------------------------------------------------------------------- */
/* Fine-tune (stub)                                                            */
/* -------------------------------------------------------------------------- */

static void cmd_finetune(int argc, char **argv) {
    const char *weights   = NULL;
    const char *data_path = NULL;
    const char *out_path  = "weights/finetuned.bin";
    int   epochs   = 5;
    int   lora_rank = 8;
    float lr       = 5e-5f;

    for (int i = 0; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--weights"))   weights   = argv[++i];
        else if (!strcmp(argv[i], "--data"))      data_path = argv[++i];
        else if (!strcmp(argv[i], "--out"))       out_path  = argv[++i];
        else if (!strcmp(argv[i], "--epochs"))    epochs    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lora-rank")) lora_rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr"))        lr        = (float)atof(argv[++i]);
    }

    if (!weights || !data_path) {
        fprintf(stderr, "error: --weights and --data are both required\n");
        exit(1);
    }

    printf("LoRA fine-tuning: weights=%s data=%s rank=%d epochs=%d lr=%.1e\n",
           weights, data_path, lora_rank, epochs, (double)lr);
    printf("Not yet implemented. See src/train.c for the training loop to extend.\n");
    (void)out_path;
}

/* -------------------------------------------------------------------------- */
/* main                                                                        */
/* -------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    if (argc < 2) usage(argv[0]);

    if      (!strcmp(argv[1], "train"))    cmd_train   (argc-2, argv+2);
    else if (!strcmp(argv[1], "infer"))    cmd_infer   (argc-2, argv+2);
    else if (!strcmp(argv[1], "finetune")) cmd_finetune(argc-2, argv+2);
    else { fprintf(stderr, "Unknown command: %s\n\n", argv[1]); usage(argv[0]); }

    return 0;
}
