#ifndef SINEW_MIND_H
#define SINEW_MIND_H

#include "sm_utils.h"
#include "sm_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SinewMind {
    u64 count;
    SM_Matrix *weights;
    SM_Matrix *biases;
    SM_Matrix *activations;
} SinewMind;

SinewMind sm_alloc(u64 *layers, u64 layer_count);
void sm_print_with(SinewMind *sm, const char *name);

#ifdef __cplusplus
}
#endif

#endif