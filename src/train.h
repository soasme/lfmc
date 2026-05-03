#ifndef TRAIN_H
#define TRAIN_H

#include "model.h"

/* --------------------------------------------------------------------------
 * Training
 * -------------------------------------------------------------------------- */

typedef struct {
    /* Data */
    const char *data_path;   /* path to .bin token file              */
    const char *out_path;    /* where to save the trained model       */

    /* Architecture (only used when training from scratch) */
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int max_seq_len;
    int ode_steps;

    /* Optimiser */
    float lr;
    float beta1;
    float beta2;
    float eps;
    float grad_clip;

    /* Schedule */
    int   max_epochs;
    int   print_every;    /* batches between loss prints   */
    int   sample_every;   /* batches between sample prints */
    int   sample_count;   /* names/sequences to sample     */
    float sample_temp;    /* sampling temperature          */
} TrainConfig;

/* Fill tc with sensible defaults */
void train_config_defaults(TrainConfig *tc);

/* Run training. model == NULL → allocate fresh from tc->* size fields.
   Returns the trained model (caller owns it). */
LFMModel *lfm_train(TrainConfig *tc, LFMModel *model);

#endif /* TRAIN_H */
