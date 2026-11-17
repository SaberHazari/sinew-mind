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

void sm_activate_softmax(SM_Matrix *mat) {
    for(u64 i = 0; i < mat->rows; ++i) {
        f64 max_value = matrix_pos_ptr_m(mat, i, 0);
        for(u64 j = 1; j < mat->cols; ++j) {
            f64 value = matrix_pos_ptr_m(mat, i, j);
            if(value > max_value) { max_value = value; }
        }

        f64 sum = 0.0;
        for(u64 j = 0; j < mat->cols; ++j) {
            f64 exp_diff = exp(matrix_pos_ptr_m(mat, i, j) - max_value);
            matrix_pos_ptr_m(mat, i, j) = exp_diff;
            sum += exp_diff;
        }

        for(u64 j = 0; j < mat->cols; ++j) {
            matrix_pos_ptr_m(mat, i, j) /= sum;
        }
    }
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

void sm_derivative_softmax(SM_Matrix *mat, const SM_Matrix *activation) {
    for(u64 i = 0; i < mat->rows; ++i) {
        f64 dot = 0.0;
        for(u64 j = 0; j < mat->cols; ++j) {
            dot += matrix_pos_ptr_m(activation, i, j) * matrix_pos_ptr_m(mat, i, j);
        }
        
        for(u64 j = 0; j < mat->cols; ++j) {
            f64 act_value = matrix_pos_ptr_m(activation, i, j);
            matrix_pos_ptr_m(mat, i, j) = act_value * (matrix_pos_ptr_m(mat, i, j) - dot);
        }
    }
}

// Cost is Half MSE here
f64 sm_cost_mse(const SM_Matrix *output, const SM_Matrix *target) {
    f64 cost = 0.0;
    for(u64 j = 0; j < output->cols; ++j) {
        f64 diff = matrix_pos_ptr_m(output, 0, j) - matrix_pos_ptr_m(target, 0, j);
        cost += 0.5 * diff * diff;
    }
    return cost;
}

f64 sm_cost_cross_entropy(const SM_Matrix *output, const SM_Matrix *target) {
    const f64 eps = 1e-15;
    f64 cost = 0.0;
    for(u64 j = 0; j < output->cols; ++j) {
        f64 output_value = matrix_pos_ptr_m(output, 0, j);
        f64 target_value = matrix_pos_ptr_m(target, 0, j);
        if(output_value < eps) { output_value = eps; }
        else if(output_value > 1.0 - eps) { output_value = 1.0 - eps; }
        cost -= target_value * log(output_value);
    }
    return cost;
}

void sm_output_delta_mse(SM_Matrix *delta, const SM_Matrix *output, 
    const SM_Matrix *target, DerivativeFn derivator) {
    for(u64 j = 0; j < output->cols; ++j) {
        matrix_pos_ptr_m(delta, 0, j) = matrix_pos_ptr_m(output, 0, j) - 
            matrix_pos_ptr_m(target, 0, j);
    }
    derivator(delta, output);
}

void sm_output_delta_softmax_ce(SM_Matrix *delta, const SM_Matrix *output, 
    const SM_Matrix *target, DerivativeFn derivator) {
    (void)derivator;
    for(u64 j = 0; j < output->cols; ++j) {
        matrix_pos_ptr_m(delta, 0, j) = matrix_pos_ptr_m(output, 0, j) - 
            matrix_pos_ptr_m(target, 0, j);
    }
}

SinewMind sm_alloc(u64 *layers, u64 layer_count, ActivationFn activator, 
    DerivativeFn derivator) {
    assert(layer_count >= 2);
    SinewMind sm;
    sm.model_layers = layer_count - 1;

    // mallocs are temporary
    sm.activations = malloc(layer_count * sizeof(*sm.activations));
    sm.weights = malloc(sm.model_layers * sizeof(*sm.weights));
    sm.biases = malloc(sm.model_layers * sizeof(*sm.biases));
    sm.deltas = malloc(sm.model_layers * sizeof(*sm.deltas));
    sm.weight_grads = malloc(sm.model_layers * sizeof(*sm.weight_grads));
    sm.bias_grads = malloc(sm.model_layers * sizeof(*sm.bias_grads));
    sm.activators = malloc(sm.model_layers * sizeof(*sm.activators));
    sm.derivators = malloc(sm.model_layers * sizeof(*sm.derivators));
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
    sm.cost_fn = sm_cost_mse;
    sm.output_delta_fn = sm_output_delta_mse;

    return sm;
}

void sm_set_layer_activation(SinewMind *sm, u64 layer, ActivationFn activator, 
    DerivativeFn derivator) {
    assert(layer < sm->model_layers);

    sm->activators[layer] = activator;
    sm->derivators[layer] = derivator;
}

void sm_set_loss(SinewMind *sm, CostFn cost_fn, OutputDeltaFn output_delta_fn) {
    sm->cost_fn = cost_fn;
    sm->output_delta_fn = output_delta_fn;
}

void sm_random(SinewMind *sm, SM_Rng *rng, f64 low, f64 high) {
    for(u64 i = 0; i < sm->model_layers; ++i) {
        matrix_random(&sm->weights[i], rng, low, high);
        matrix_random(&sm->biases[i], rng, low, high);
    }
}

void sm_forward(SinewMind *sm) {
    for(u64 l = 0; l < sm->model_layers; ++l) {
        matrix_mul(&sm->activations[l + 1], &sm->activations[l], &sm->weights[l]);
        matrix_add(&sm->activations[l + 1], &sm->biases[l]);
        sm->activators[l](&sm->activations[l + 1]);
    }
}

SM_DataSplit sm_train_test_split(SM_Matrix *input, SM_Matrix *target, 
    SM_Rng *rng, f64 test_fraction) {
    assert(input->rows >= 2);
    assert(input->rows == target->rows);
    assert(test_fraction > 0.0 && test_fraction < 1.0);

    u64 total_size = input->rows;
    u64 test_size = (u64)((f64)total_size * test_fraction + 0.5);
    if(test_size < 1) { test_size = 1; }
    if(test_size >= total_size) { test_size = total_size - 1; }
    u64 train_size = total_size - test_size;

    SM_DataSplit split;
    split.x_train = matrix_alloc(train_size, input->cols);
    split.y_train = matrix_alloc(train_size, target->cols);
    split.x_test = matrix_alloc(test_size, input->cols);
    split.y_test = matrix_alloc(test_size, target->cols);

    u64 *idx_buff = malloc(total_size * sizeof(*idx_buff));
    assert(idx_buff);
    for(u64 i = 0; i < total_size; ++i) { idx_buff[i] = i; }
    rng_shuffle(rng, idx_buff, total_size);

    for(u64 i = 0; i < train_size; ++i) {
        SM_Matrix x_data_i = matrix_row_view_m(*input, idx_buff[i], 1);
        SM_Matrix y_data_i = matrix_row_view_m(*target, idx_buff[i], 1);
        SM_Matrix x_train_i = matrix_row_view_m(split.x_train, i, 1);
        SM_Matrix y_train_i = matrix_row_view_m(split.y_train, i, 1);
        matrix_copy_m(x_train_i, x_data_i);
        matrix_copy_m(y_train_i, y_data_i);
    }
    for(u64 j = 0; j < test_size; ++j) {
        SM_Matrix x_data_j = matrix_row_view_m(*input, idx_buff[train_size + j], 1);
        SM_Matrix y_data_j = matrix_row_view_m(*target, idx_buff[train_size + j], 1);
        SM_Matrix x_test_j = matrix_row_view_m(split.x_test, j, 1);
        SM_Matrix y_test_j = matrix_row_view_m(split.y_test, j, 1);
        matrix_copy_m(x_test_j, x_data_j);
        matrix_copy_m(y_test_j, y_data_j);
    }

    free(idx_buff);
    return split;
}

f64 sm_cost(SinewMind *sm, SM_Matrix *input, SM_Matrix *target) {
    assert(input->rows == target->rows);
    assert(input->cols == sm->activations[0].cols);
    assert(target->cols == sm->activations[sm->model_layers].cols);

    f64 cost = 0.0;
    for(u64 s = 0; s < input->rows; ++s) {
        SM_Matrix x = matrix_row_view_m(*input, s, 1);
        SM_Matrix y = matrix_row_view_m(*target, s, 1);

        matrix_copy_m(sm->activations[0], x);
        sm_forward(sm);

        cost += sm->cost_fn(&sm->activations[sm->model_layers], &y);
    }

    return cost / input->rows;
}

void sm_zero_grads(SinewMind *sm) {
    for(u64 l = 0; l < sm->model_layers; ++l) {
        for(u64 i = 0; i < sm->weight_grads[l].rows; ++i) {
            for(u64 j = 0; j < sm->weight_grads[l].cols; ++j) {
                matrix_pos_m(sm->weight_grads[l], i, j) = 0.0;
            }
        }

        for(u64 j = 0; j < sm->bias_grads[l].cols; ++j) {
            matrix_pos_m(sm->bias_grads[l], 0, j) = 0.0;
        }
    }
}

void sm_backprop_accumulate(SinewMind *sm, SM_Matrix *input, SM_Matrix *target) {
    assert(input->rows == target->rows);
    assert(input->cols == sm->activations[0].cols);
    assert(target->cols == sm->activations[sm->model_layers].cols);

    for(u64 s = 0; s < input->rows; ++s) {
        SM_Matrix x = matrix_row_view_m(*input, s, 1);
        SM_Matrix y = matrix_row_view_m(*target, s, 1);

        matrix_copy_m(sm->activations[0], x);
        sm_forward(sm);

        SM_Matrix *final_delta = &sm->deltas[sm->model_layers - 1];
        sm->output_delta_fn(final_delta, &sm->activations[sm->model_layers], &y, 
            sm->derivators[sm->model_layers - 1]);

        for(u64 l = sm->model_layers; l >= 1; --l) {
            SM_Matrix *activation = &sm->activations[l - 1];
            SM_Matrix *weight = &sm->weights[l - 1];
            SM_Matrix *bias = &sm->biases[l - 1];
            SM_Matrix *delta = &sm->deltas[l - 1];
            SM_Matrix *weight_grad = &sm->weight_grads[l - 1];
            SM_Matrix *bias_grad = &sm->bias_grads[l - 1];

            for(u64 i = 0; i < weight->rows; ++i) {
                f64 prev_act_value = matrix_pos_ptr_m(activation, 0, i);
                for(u64 j = 0; j < weight->cols; ++j) {
                    matrix_pos_ptr_m(weight_grad, i, j) += prev_act_value * 
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

void sm_backprop(SinewMind *sm, SM_Matrix *input, SM_Matrix *target) {
    sm_zero_grads(sm);
    sm_backprop_accumulate(sm, input, target);
}

void sm_step(SinewMind *sm, f64 learning_rate, f64 batch_size) {
    f64 scale = learning_rate / batch_size;
    for(u64 l = 0; l < sm->model_layers; ++l) {
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

void sm_train(SinewMind *sm, SM_Matrix *x_train, SM_Matrix *y_train, SM_Rng *rng, 
    u64 epochs, u64 batch_size, f64 learning_rate) {
    assert(x_train->rows == y_train->rows);
    assert(x_train->cols == sm->activations[0].cols);
    assert(y_train->cols == sm->activations[sm->model_layers].cols);
    assert(batch_size >= 1);

    u64 train_size = x_train->rows;
    if(batch_size > train_size) { batch_size = train_size; }

    u64 *idx_buff = malloc(train_size * sizeof(*idx_buff));
    assert(idx_buff);

    for(u64 epoch = 0; epoch < epochs; ++epoch) {
        for(u64 i = 0; i < train_size; ++i) { idx_buff[i] = i; }
        rng_shuffle(rng, idx_buff, train_size);

        for(u64 current = 0; current < train_size; current += batch_size) {
            u64 left = train_size - current;
            if(left > batch_size) { left = batch_size; }

            sm_zero_grads(sm);
            
            for(u64 j = 0; j < left; ++j) {
                SM_Matrix x = matrix_row_view_m(*x_train, idx_buff[current + j], 1);
                SM_Matrix y = matrix_row_view_m(*y_train, idx_buff[current + j], 1);
                sm_backprop_accumulate(sm, &x, &y);
            }

            sm_step(sm, learning_rate, (f64)left);
        }
    }

    free(idx_buff);
}

static bool sm_output_is_correct_(const SM_Matrix *output, const SM_Matrix *target, 
    u64 output_dim, f64 threshold) {
    for(u64 o = 0; o < output_dim; ++o) {
        f64 output_value = matrix_pos_ptr_m(output, 0, o);
        f64 target_value = matrix_pos_ptr_m(target, 0, o);

        if(fabs(output_value - target_value) > threshold) {
            return false;
        }
    }
    return true;
}

static bool sm_output_is_correct_binary_(const SM_Matrix *output, const SM_Matrix *target, 
    u64 output_dim, f64 threshold) {
    for(u64 o = 0; o < output_dim; ++o) {
        f64 output_value = matrix_pos_ptr_m(output, 0, o);
        f64 target_value = matrix_pos_ptr_m(target, 0, o);

        if((output_value >= threshold) != (target_value >= threshold)) {
            return false;
        }
    }
    return true;
}

SM_Metrics sm_test(SinewMind *sm, SM_Matrix *x_test, SM_Matrix *y_test, 
    f64 threshold, bool is_binary) {
    assert(x_test->rows > 0);
    assert(x_test->rows == y_test->rows);
    assert(x_test->cols == sm->activations[0].cols);
    assert(y_test->cols == sm->activations[sm->model_layers].cols);

    u64 test_size = x_test->rows;
    u64 output_dim = y_test->cols;

    SM_Metrics metrics = {0};
    metrics.total = test_size;

    f64 total_cost = 0.0;
    f64 total_sq = 0.0;
    f64 total_abs = 0.0;

    for(u64 s = 0; s < test_size; ++s) {
        SM_Matrix x = matrix_row_view_m(*x_test, s, 1);
        SM_Matrix y = matrix_row_view_m(*y_test, s, 1);

        matrix_copy_m(sm->activations[0], x);
        sm_forward(sm);

        SM_Matrix *output = &sm->activations[sm->model_layers];
        total_cost += sm->cost_fn(output, &y);

        for(u64 o = 0; o < output_dim; ++o) {
            f64 diff = matrix_pos_ptr_m(output, 0, o) - matrix_pos_m(y, 0, o);
            total_sq += diff * diff;
            total_abs += fabs(diff);
        }

        bool correct;
        if(is_binary) {
            correct = sm_output_is_correct_binary_(output, &y, output_dim, threshold);
        } else {
            correct = sm_output_is_correct_(output, &y, output_dim, threshold);
        }
        if(correct) { metrics.correct++; }
    }

    metrics.cost = total_cost / (f64)metrics.total;
    metrics.mse = total_sq / (f64)(metrics.total * output_dim);
    metrics.rmse = sqrt(metrics.mse);
    metrics.mae = total_abs / (f64)(metrics.total * output_dim);
    metrics.accuracy = (f64)metrics.correct / (f64)metrics.total;

    return metrics;
}

void sm_print(SinewMind *sm, const char *name) {
    char buffer[256];
    printf("%s = [\n", name);
    for(u64 i = 0; i < sm->model_layers; ++i) {
        snprintf(buffer, sizeof(buffer), "weights[%" PRIu64 "]", i);
        matrix_print(&sm->weights[i], buffer, 2);
        snprintf(buffer, sizeof(buffer), "biases[%" PRIu64 "]", i);
        matrix_print(&sm->biases[i], buffer, 2);
    }
    printf("]\n");
}

void sm_metrics_print(SM_Metrics *metrics) {
    printf("\n\n<----------MODEL METRICS---------->\n");
    printf("Total Test Samples = %" PRIu64 "\n", metrics->total);
    printf("Correctly Predicted = %" PRIu64 "\n", metrics->correct);
    printf("Accuracy = %.10f%%\n", metrics->accuracy * 100.0);
    printf("MSE = %.10f\n", metrics->mse);
    printf("RMSE = %.10f\n", metrics->rmse);
    printf("MAE = %.10f\n", metrics->mae);
    printf("Cost = %.10f\n\n", metrics->cost);
}

void sm_data_split_free(SM_DataSplit *split) {
    matrix_free(&split->x_train);
    matrix_free(&split->y_train);
    matrix_free(&split->x_test);
    matrix_free(&split->y_test);
}

void sm_free(SinewMind *sm) {
    matrix_free(&sm->activations[sm->model_layers]);
    for(u64 i = 0; i < sm->model_layers; ++i) {
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
    sm->model_layers = 0;
}
