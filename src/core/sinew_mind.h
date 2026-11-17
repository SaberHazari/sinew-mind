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
typedef f64 (*CostFn)(const SM_Matrix *output, const SM_Matrix *target);
typedef void (*OutputDeltaFn)(SM_Matrix *delta, const SM_Matrix *output, 
    const SM_Matrix *target, DerivativeFn derivator);

typedef struct SinewMind {
    u64 model_layers;
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

typedef struct SM_DataSplit {
    SM_Matrix x_train;
    SM_Matrix y_train;
    SM_Matrix x_test;
    SM_Matrix y_test;
} SM_DataSplit;

typedef struct SM_Metrics {
    f64 cost;
    f64 mse;
    f64 rmse;
    f64 mae;
    f64 accuracy;
    u64 correct;
    u64 total;
} SM_Metrics;

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

f64 sm_cost_mse(const SM_Matrix *output, const SM_Matrix *target);
f64 sm_cost_cross_entropy(const SM_Matrix *output, const SM_Matrix *target);

void sm_output_delta_mse(SM_Matrix *delta, const SM_Matrix *output, 
    const SM_Matrix *target, DerivativeFn derivator);
void sm_output_delta_softmax_ce(SM_Matrix *delta, const SM_Matrix *output, 
    const SM_Matrix *target, DerivativeFn derivator);

SinewMind sm_alloc(u64 *layers, u64 layer_count, ActivationFn activator, 
    DerivativeFn derivator);
void sm_set_layer_activation(SinewMind *sm, u64 layer, ActivationFn activator, 
    DerivativeFn derivator);
void sm_set_loss(SinewMind *sm, CostFn cost_fn, OutputDeltaFn output_delta_fn);
void sm_random(SinewMind *sm, SM_Rng *rng, f64 low, f64 high);
void sm_forward(SinewMind *sm);
SM_DataSplit sm_train_test_split(SM_Matrix *input, SM_Matrix *target, 
    SM_Rng *rng, f64 test_fraction);
f64 sm_cost(SinewMind *sm, SM_Matrix *input, SM_Matrix *target);
void sm_zero_grads(SinewMind *sm);
void sm_backprop_accumulate(SinewMind *sm, SM_Matrix *input, SM_Matrix *target);
void sm_backprop(SinewMind *sm, SM_Matrix *input, SM_Matrix *target);
void sm_step(SinewMind *sm, f64 learning_rate, f64 batch_size);
void sm_train(SinewMind *sm, SM_Matrix *x_train, SM_Matrix *y_train, SM_Rng *rng, 
    u64 epochs, u64 batch_size, f64 learning_rate);
SM_Metrics sm_test(SinewMind *sm, SM_Matrix *x_test, SM_Matrix *y_test, 
    f64 threshold, bool is_binary);
void sm_print(SinewMind *sm, const char *name);
void sm_metrics_print(SM_Metrics *metrics);
void sm_data_split_free(SM_DataSplit *split);
void sm_free(SinewMind *sm);

#ifdef __cplusplus
}
#endif

#endif