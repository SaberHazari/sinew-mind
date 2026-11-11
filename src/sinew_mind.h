#ifndef SINEW_MIND_H
#define SINEW_MIND_H

#include "sm_utils.h"
#include "sm_rng.h"
#include "sm_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SinewMind {
    u64 count;
    SM_Matrix *weights;
    SM_Matrix *biases;
    SM_Matrix *activations;
    ActivationFn activator;
} SinewMind;

SinewMind sm_alloc(u64 *layers, u64 layer_count, ActivationFn activator);
void sm_random(SinewMind *sm, SM_Rng *rng, f64 lo, f64 hi);
void sm_forward(SinewMind *sm);
f64 sm_cost(SinewMind *sm, SM_Matrix *ti, SM_Matrix *to);
void sm_print(SinewMind *sm, const char *name);

#ifdef __cplusplus
}
#endif

#endif