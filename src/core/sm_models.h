#ifndef SM_MODELS_H
#define SM_MODELS_H

#include "sinew_mind.h"

#ifdef __cplusplus
extern "C" {
#endif

SinewMind sm_linear_regressor(u64 input_dim, u64 output_dim, SM_Rng *rng);
SinewMind sm_logistic_regressor(u64 input_dim, u64 output_dim, SM_Rng *rng);
SinewMind sm_softmax_regressor(u64 input_dim, u64 output_dim, SM_Rng *rng);

SinewMind sm_mlp_regressor(u64 *hidden_layers, u64 hidden_count, u64 input_dim, 
    u64 output_dim, ActivationFn activator, DerivativeFn derivator, SM_Rng *rng);

SinewMind sm_mlp_regressor_default(u64 *hidden_layers, u64 hidden_count, u64 input_dim, 
    u64 output_dim, SM_Rng *rng);

#ifdef __cplusplus
}
#endif

#endif