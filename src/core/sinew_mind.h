#ifndef SINEW_MIND_H
#define SINEW_MIND_H

#include "sm_utils.h"
#include "sm_rng.h"
#include "sm_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ActivationFn)(SM_Matrix *mat);
typedef void (*DerivativeFn)(SM_Matrix *mat, const SM_Matrix *activation);
typedef f64 (*CostFn)(const SM_Matrix *pred, const SM_Matrix *target);
typedef void (*OutputDeltaFn)(SM_Matrix *delta, const SM_Matrix *pred, 
    const SM_Matrix *target, DerivativeFn derivator);

typedef struct SinewMind {
    u64 count;
    SM_Matrix *activations;
    SM_Matrix *weights;
    SM_Matrix *biases;
    SM_Matrix *deltas;
    SM_Matrix *weight_grads;
    SM_Matrix *bias_grads;
    ActivationFn *activators;
    DerivativeFn *derivators;
    CostFn cost_fn;
    OutputDeltaFn output_delta_fn;
} SinewMind;

void sm_activate_sigmoid(SM_Matrix *mat);
void sm_activate_tanh(SM_Matrix *mat);
void sm_activate_relu(SM_Matrix *mat);
void sm_activate_linear(SM_Matrix *mat);
void sm_activate_softmax(SM_Matrix *mat);

void sm_derivative_sigmoid(SM_Matrix *mat, const SM_Matrix *activation);
void sm_derivative_tanh(SM_Matrix *mat, const SM_Matrix *activation);
void sm_derivative_relu(SM_Matrix *mat, const SM_Matrix *activation);
void sm_derivative_linear(SM_Matrix *mat, const SM_Matrix *activation);
void sm_derivative_softmax(SM_Matrix *mat, const SM_Matrix *activation);

f64 sm_cost_mse(const SM_Matrix *pred, const SM_Matrix *target);
f64 sm_cost_cross_entropy(const SM_Matrix *pred, const SM_Matrix *target);

void sm_output_delta_mse(SM_Matrix *delta, const SM_Matrix *pred, 
    const SM_Matrix *target, DerivativeFn derivator);
void sm_output_delta_softmax_ce(SM_Matrix *delta, const SM_Matrix *pred, 
    const SM_Matrix *target, DerivativeFn derivator);

SinewMind sm_alloc(u64 *layers, u64 layer_count, ActivationFn activator, 
    DerivativeFn derivator);
void sm_set_layer_activation(SinewMind *sm, u64 layer, ActivationFn activator, 
    DerivativeFn derivator);
void sm_set_loss(SinewMind *sm, CostFn cost_fn, OutputDeltaFn output_delta_fn);
void sm_random(SinewMind *sm, SM_Rng *rng, f64 low, f64 high);
void sm_forward(SinewMind *sm);
f64 sm_cost(SinewMind *sm, SM_Matrix *input, SM_Matrix *output);
void sm_backprop(SinewMind *sm, SM_Matrix *input, SM_Matrix *output);
void sm_step(SinewMind *sm, f64 learning_rate, f64 batch_size);
void sm_print(SinewMind *sm, const char *name);
void sm_free(SinewMind *sm);

#ifdef __cplusplus
}
#endif

#endif