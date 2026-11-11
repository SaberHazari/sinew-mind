#include "sinew_mind.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

SinewMind sm_alloc(u64 *layers, u64 layer_count, ActivationFn activator) {
    assert(layer_count >= 2);
    SinewMind sm;
    sm.count = layer_count - 1;

    // mallocs are temporary
    sm.weights = malloc(sm.count * sizeof(*sm.weights));
    sm.biases = malloc(sm.count * sizeof(*sm.biases));
    sm.activations = malloc(layer_count * sizeof(*sm.activations));
    assert(sm.weights && sm.biases && sm.activations);

    sm.activations[0] = matrix_alloc(1, layers[0]);
    for(u64 i = 1; i < layer_count; ++i) {
        sm.weights[i - 1] = matrix_alloc(sm.activations[i - 1].cols, layers[i]);
        sm.biases[i - 1] = matrix_alloc(1, layers[i]);
        sm.activations[i] = matrix_alloc(1, layers[i]);
    }
    sm.activator = activator;

    return sm;
}

void sm_random(SinewMind *sm, SM_Rng *rng, f64 lo, f64 hi) {
    for(u64 i = 0; i < sm->count; ++i) {
        matrix_random(&sm->weights[i], rng, lo, hi);
        matrix_random(&sm->biases[i], rng, lo, hi);
    }
}

void sm_forward(SinewMind *sm) {
    for(u64 i = 0; i < sm->count; ++i) {
        matrix_mul(&sm->activations[i + 1], &sm->activations[i], &sm->weights[i]);
        matrix_add(&sm->activations[i + 1], &sm->biases[i]);
        matrix_activate(&sm->activations[i + 1], sm->activator);
    }
}

f64 sm_cost(SinewMind *sm, SM_Matrix *ti, SM_Matrix *to) {
    assert(ti->rows == to->rows);
    assert(to->cols == sm->activations[sm->count].cols);

    f64 cost = 0.0;
    for(u64 i = 0; i < ti->rows; ++i) {
        SM_Matrix x = matrix_row_view_m(*ti, i, 1);
        SM_Matrix y = matrix_row_view_m(*to, i, 1);

        matrix_copy_m(sm->activations[0], x);
        sm_forward(sm);

        for(u64 j = 0; j < to->cols; ++j) {
            f64 diff = matrix_pos_m(sm->activations[sm->count], 0, j) - matrix_pos_m(y, 0, j);
            cost += diff * diff;
        }
    }

    return cost / ti->rows;
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
