#include "sm_models.h"
#include <stdlib.h>

static f64 sm_models_init_scale_(u64 fan_in) {
    return fan_in ? 1.0 / sqrt((f64)fan_in) : 0.01;
}

static u64 sm_models_max_fan_(u64 input_dim, u64 *hidden, u64 hidden_count) {
    u64 max = input_dim;
    for(u64 i = 0; i < hidden_count; ++i) {
        if(hidden[i] > max) { max = hidden[i]; }
    }
    return max;
}

static SinewMind sm_models_build_(u64 input_dim, u64 *hidden, u64 hidden_count, 
    u64 output_dim, ActivationFn activator, DerivativeFn derivator) {
    u64 layer_count = hidden_count + 2;
    u64 *layers = malloc(layer_count * sizeof(*layers));
    assert(layers);

    layers[0] = input_dim;
    for(u64 i = 0; i < hidden_count; ++i) { layers[i + 1] = hidden[i]; }
    layers[layer_count - 1] = output_dim;

    SinewMind sm = sm_alloc(layers, layer_count, activator, derivator);
    free(layers);
    return sm;
}

SinewMind sm_linear_regressor(u64 input_dim, u64 output_dim, SM_Rng *rng) {
    assert(input_dim >= 1 && output_dim >= 1);

    SinewMind sm = sm_models_build_(input_dim, NULL, 0, output_dim, 
        sm_activate_linear, sm_derivative_linear);
    sm_set_loss(&sm, sm_cost_mse, sm_output_delta_mse);

    f64 scale = sm_models_init_scale_(input_dim);
    sm_random(&sm, rng, -scale, scale);
    return sm;
}

SinewMind sm_logistic_regressor(u64 input_dim, u64 output_dim, SM_Rng *rng) {
    assert(input_dim >= 1 && output_dim >= 1);

    SinewMind sm = sm_models_build_(input_dim, NULL, 0, output_dim, 
        sm_activate_sigmoid, sm_derivative_sigmoid);
    sm_set_loss(&sm, sm_cost_binary_cross_entropy, sm_output_delta_sigmoid_bce);

    f64 scale = sm_models_init_scale_(input_dim);
    sm_random(&sm, rng, -scale, scale);
    return sm;
}

SinewMind sm_softmax_regressor(u64 input_dim, u64 output_dim, SM_Rng *rng) {
    assert(input_dim >= 1 && output_dim >= 2);
    SinewMind sm = sm_models_build_(input_dim, NULL, 0, output_dim, 
        sm_activate_softmax, sm_derivative_softmax);
    sm_set_loss(&sm, sm_cost_cross_entropy, sm_output_delta_softmax_ce);

    f64 scale = sm_models_init_scale_(input_dim);
    sm_random(&sm, rng, -scale, scale);
    return sm;
}

SinewMind sm_mlp_regressor(u64 *hidden_layers, u64 hidden_count, u64 input_dim, 
    u64 output_dim, ActivationFn activator, DerivativeFn derivator, SM_Rng *rng) {
    assert(input_dim >= 1 && output_dim >= 1);

    SinewMind sm = sm_models_build_(input_dim, hidden_layers, hidden_count, 
        output_dim, activator, derivator);
    sm_set_layer_activation(&sm, sm.model_layers - 1, sm_activate_linear, 
        sm_derivative_linear);
    sm_set_loss(&sm, sm_cost_mse, sm_output_delta_mse);

    f64 scale = sm_models_init_scale_(
        sm_models_max_fan_(input_dim, hidden_layers, hidden_count));
    sm_random(&sm, rng, -scale, scale);
    return sm;
}

SinewMind sm_mlp_regressor_default(u64 *hidden_layers, u64 hidden_count, u64 input_dim, 
    u64 output_dim, SM_Rng *rng) {
    return sm_mlp_regressor(hidden_layers, hidden_count, input_dim, output_dim, 
        sm_activate_relu, sm_derivative_relu, rng);
}