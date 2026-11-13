#include "sinew_mind.h"
#include <stdlib.h>
#include <stdio.h>

void sm_activate_sigmoid(SM_Matrix *mat) {
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            matrix_pos_ptr_m(mat, i, j) = 1.0 / (1.0 + exp(-matrix_pos_ptr_m(mat, i, j)));
        }
    }
}

void sm_activate_tanh(SM_Matrix *mat) {
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            matrix_pos_ptr_m(mat, i, j) = tanh(matrix_pos_ptr_m(mat, i, j));
        }
    }
}

void sm_activate_relu(SM_Matrix *mat) {
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            if(matrix_pos_ptr_m(mat, i, j) < 0.0) {
                matrix_pos_ptr_m(mat, i, j) = 0.0;
            }
        }
    }
}

void sm_activate_linear(SM_Matrix *mat) {
    (void)mat;
}

void sm_derivative_sigmoid(SM_Matrix *mat, const SM_Matrix *activation) {
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            f64 act_value = matrix_pos_ptr_m(activation, i, j);
            matrix_pos_ptr_m(mat, i, j) *= act_value * (1.0 - act_value);
        }
    }
}

void sm_derivative_tanh(SM_Matrix *mat, const SM_Matrix *activation) {
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            f64 act_value = matrix_pos_ptr_m(activation, i, j);
            matrix_pos_ptr_m(mat, i, j) *= 1.0 - act_value * act_value;
        }
    }
}

void sm_derivative_relu(SM_Matrix *mat, const SM_Matrix *activation) {
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            if(matrix_pos_ptr_m(activation, i, j) <= 0.0) {
                matrix_pos_ptr_m(mat, i, j) = 0.0;
            }
        }
    }
}

void sm_derivative_linear(SM_Matrix *mat, const SM_Matrix *activation) {
    (void)mat; (void)activation;
}

SinewMind sm_alloc(u64 *layers, u64 layer_count, ActivationFn activator, 
    DerivativeFn derivator) {
    assert(layer_count >= 2);
    SinewMind sm;
    sm.count = layer_count - 1;

    // mallocs are temporary
    sm.activations = malloc(layer_count * sizeof(*sm.activations));
    sm.weights = malloc(sm.count * sizeof(*sm.weights));
    sm.biases = malloc(sm.count * sizeof(*sm.biases));
    sm.deltas = malloc(sm.count * sizeof(*sm.deltas));
    sm.weight_grads = malloc(sm.count * sizeof(*sm.weight_grads));
    sm.bias_grads = malloc(sm.count * sizeof(*sm.bias_grads));
    sm.activators = malloc(sm.count * sizeof(*sm.activators));
    sm.derivators = malloc(sm.count * sizeof(*sm.derivators));
    assert(sm.activations && sm.weights && sm.biases && 
        sm.deltas && sm.weight_grads && sm.bias_grads && 
        sm.activators && sm.derivators);

    sm.activations[0] = matrix_alloc(1, layers[0]);
    for(u64 l = 1; l < layer_count; ++l) {
        sm.activations[l] = matrix_alloc(1, layers[l]);
        sm.weights[l - 1] = matrix_alloc(layers[l - 1], layers[l]);
        sm.biases[l - 1] = matrix_alloc(1, layers[l]);
        sm.deltas[l - 1] = matrix_alloc(1, layers[l]);
        sm.weight_grads[l - 1] = matrix_alloc(layers[l - 1], layers[l]);
        sm.bias_grads[l - 1] = matrix_alloc(1, layers[l]);
        sm.activators[l - 1] = activator;
        sm.derivators[l - 1] = derivator;
    }

    return sm;
}

void sm_set_layer_activation(SinewMind *sm, u64 layer, ActivationFn activator, 
    DerivativeFn derivator) {
    assert(layer < sm->count);

    sm->activators[layer] = activator;
    sm->derivators[layer] = derivator;
}

void sm_random(SinewMind *sm, SM_Rng *rng, f64 low, f64 high) {
    for(u64 i = 0; i < sm->count; ++i) {
        matrix_random(&sm->weights[i], rng, low, high);
        matrix_random(&sm->biases[i], rng, low, high);
    }
}

void sm_forward(SinewMind *sm) {
    for(u64 l = 0; l < sm->count; ++l) {
        matrix_mul(&sm->activations[l + 1], &sm->activations[l], &sm->weights[l]);
        matrix_add(&sm->activations[l + 1], &sm->biases[l]);
        sm->activators[l](&sm->activations[l + 1]);
    }
}

// Cost is half MSE here
f64 sm_cost(SinewMind *sm, SM_Matrix *ti, SM_Matrix *to) {
    assert(ti->rows == to->rows);
    assert(ti->cols == sm->activations[0].cols);
    assert(to->cols == sm->activations[sm->count].cols);

    f64 cost = 0.0;
    for(u64 s = 0; s < ti->rows; ++s) {
        SM_Matrix x = matrix_row_view_m(*ti, s, 1);
        SM_Matrix y = matrix_row_view_m(*to, s, 1);

        matrix_copy_m(sm->activations[0], x);
        sm_forward(sm);

        for(u64 o = 0; o < to->cols; ++o) {
            f64 diff = matrix_pos_m(sm->activations[sm->count], 0, o) - matrix_pos_m(y, 0, o);
            cost += 0.5 * diff * diff;
        }
    }

    return cost / ti->rows;
}

void sm_backprop(SinewMind *sm, SM_Matrix *ti, SM_Matrix *to) {
    assert(ti->rows == to->rows);
    assert(ti->cols == sm->activations[0].cols);
    assert(to->cols == sm->activations[sm->count].cols);

    for(u64 l = 0; l < sm->count; ++l) {
        for(u64 i = 0; i < sm->weight_grads[l].rows; ++i) {
            for(u64 j = 0; j < sm->weight_grads[l].cols; ++j) {
                matrix_pos_m(sm->weight_grads[l], i, j) = 0.0;
            }
        }

        for(u64 j = 0; j < sm->bias_grads[l].cols; ++j) {
            matrix_pos_m(sm->bias_grads[l], 0, j) = 0.0;
        }
    }

    for(u64 s = 0; s < ti->rows; ++s) {
        SM_Matrix x = matrix_row_view_m(*ti, s, 1);
        SM_Matrix y = matrix_row_view_m(*to, s, 1);

        matrix_copy_m(sm->activations[0], x);
        sm_forward(sm);

        SM_Matrix *final_delta = &sm->deltas[sm->count - 1];
        for(u64 o = 0; o < to->cols; ++o) {
            matrix_pos_ptr_m(final_delta, 0, o) = matrix_pos_m(sm->activations[sm->count], 0, o) - 
                matrix_pos_m(y, 0, o);
        }
        sm->derivators[sm->count - 1](final_delta, &sm->activations[sm->count]);

        for(u64 l = sm->count; l >= 1; --l) {
            SM_Matrix *activation = &sm->activations[l - 1];
            SM_Matrix *weight = &sm->weights[l - 1];
            SM_Matrix *bias = &sm->biases[l - 1];
            SM_Matrix *delta = &sm->deltas[l - 1];
            SM_Matrix *weight_grad = &sm->weight_grads[l - 1];
            SM_Matrix *bias_grad = &sm->bias_grads[l - 1];

            for(u64 i = 0; i < weight->rows; ++i) {
                f64 prev_activation_value = matrix_pos_ptr_m(activation, 0, i);
                for(u64 j = 0; j < weight->cols; ++j) {
                    matrix_pos_ptr_m(weight_grad, i, j) += prev_activation_value * 
                        matrix_pos_ptr_m(delta, 0, j);
                }
            }

            for(u64 j = 0; j < bias->cols; ++j) {
                matrix_pos_ptr_m(bias_grad, 0, j) += matrix_pos_ptr_m(delta, 0, j);
            }

            if(l > 1) {
                SM_Matrix *prev_delta = &sm->deltas[l - 2];
                for(u64 i = 0; i < weight->rows; ++i) {
                    f64 sum = 0.0;
                    for(u64 j = 0; j < weight->cols; ++j) {
                        sum += matrix_pos_ptr_m(weight, i, j) * matrix_pos_ptr_m(delta, 0, j);
                    }
                    matrix_pos_ptr_m(prev_delta, 0, i) = sum;
                }
                sm->derivators[l - 2](prev_delta, &sm->activations[l - 1]);
            }
        }
    }
}

// Batch size must equal the training samples used in backpropagation and finding cost
void sm_step(SinewMind *sm, f64 learning_rate, f64 batch_size) {
    f64 scale = learning_rate / batch_size;
    for(u64 l = 0; l < sm->count; ++l) {
        for(u64 i = 0; i < sm->weights[l].rows; ++i) {
            for(u64 j = 0; j < sm->weights[l].cols; ++j) {
                matrix_pos_m(sm->weights[l], i, j) -= scale * 
                    matrix_pos_m(sm->weight_grads[l], i, j);
            }
        }

        for(u64 j = 0; j < sm->biases[l].cols; ++j) {
            matrix_pos_m(sm->biases[l], 0, j) -= scale *
                matrix_pos_m(sm->bias_grads[l], 0, j);
        }
    }
}

void sm_print(SinewMind *sm, const char *name) {
    char buffer[256];
    printf("%s = [\n", name);
    for(u64 i = 0; i < sm->count; ++i) {
        snprintf(buffer, sizeof(buffer), "weights[%" PRIu64 "]", i);
        matrix_print(&sm->weights[i], buffer, 2);
        snprintf(buffer, sizeof(buffer), "biases[%" PRIu64 "]", i);
        matrix_print(&sm->biases[i], buffer, 2);
    }
    printf("]\n");
}

void sm_free(SinewMind *sm) {
    matrix_free(&sm->activations[sm->count]);
    for(u64 i = 0; i < sm->count; ++i) {
        matrix_free(&sm->activations[i]);
        matrix_free(&sm->weights[i]);
        matrix_free(&sm->biases[i]);
        matrix_free(&sm->deltas[i]);
        matrix_free(&sm->weight_grads[i]);
        matrix_free(&sm->bias_grads[i]);
    }
    free(sm->activations); free(sm->weights); free(sm->biases);
    free(sm->deltas); free(sm->weight_grads); free(sm->bias_grads); 
    free(sm->activators); free(sm->derivators);
    sm->count = 0;
}
