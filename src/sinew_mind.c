#include "sinew_mind.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

SinewMind sm_alloc(u64 *layers, u64 layer_count) {
    assert(layer_count > 0);
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

    return sm;
}

void sm_print_with(SinewMind *sm, const char *name) {
    char buffer[256];
    printf("%s = [\n", name);
    for(u64 i = 0; i < sm->count; ++i) {
        snprintf(buffer, sizeof(buffer), "weights[%lu]", i);
        matrix_print_with(&sm->weights[i], buffer, 2);
        snprintf(buffer, sizeof(buffer), "biases[%lu]", i);
        matrix_print_with(&sm->biases[i], buffer, 2);
    }
    printf("]\n");
}