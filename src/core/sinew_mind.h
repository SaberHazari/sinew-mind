#ifndef SINEW_MIND_H
#define SINEW_MIND_H

#include "sm_utils.h"
#include "sm_rng.h"
#include "sm_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef f64 (*DerivativeFn)(f64 x);

typedef struct SinewMind {
    u64 count;
    SM_Matrix *activations;
    SM_Matrix *weights;
    SM_Matrix *biases;
    SM_Matrix *deltas;
    SM_Matrix *weight_grads;
    SM_Matrix *bias_grads;
    ActivationFn activator;
    DerivativeFn derivator;
} SinewMind;

SinewMind sm_alloc(u64 *layers, u64 layer_count, ActivationFn activator, 
    DerivativeFn derivator);
void sm_random(SinewMind *sm, SM_Rng *rng, f64 lo, f64 hi);
void sm_forward(SinewMind *sm);
f64 sm_cost(SinewMind *sm, SM_Matrix *ti, SM_Matrix *to);
void sm_backprop(SinewMind *sm, SM_Matrix *ti, SM_Matrix *to);
void sm_step(SinewMind *sm, f64 learning_rate, f64 batch_size);
void sm_print(SinewMind *sm, const char *name);
void sm_free(SinewMind *sm);

#ifdef __cplusplus
}
#endif

#endif