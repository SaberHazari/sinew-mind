#include "core/sinew_mind.h"
#include <stdio.h>

#define BITS 2

int main(void) {
    SM_Rng rng;
    u64 seed = rng_seed(&rng);
    printf("\n\nSeed = %" PRIu64 "\n", seed);

    u64 n = (1 << BITS);
    u64 rows = n * n;
    SM_Matrix ti = matrix_alloc(rows, 2 * BITS);
    SM_Matrix to = matrix_alloc(rows, BITS + 1);
    for(u64 i = 0; i < ti.rows; ++i) {
        u64 x = i / n;
        u64 y = i % n;
        u64 z = x + y;
        matrix_pos_m(to, i, BITS) = (f64)(z >= n);
        for(u64 j = 0; j < BITS; ++j) {
            matrix_pos_m(ti, i, j) = (f64)((x >> j) & 1);
            matrix_pos_m(ti, i, j + BITS) = (f64)((y >> j) & 1);
            matrix_pos_m(to, i, j) = (f64)((z >> j) & 1);
        }
    }
    printf("\n--------------------------------------------\n\n");
    matrix_print(&ti, "Input", 0);
    printf("\n--------------------------------------------\n\n");

    u64 layers[] = {2 * BITS, 2 * BITS * BITS, 2 * BITS, BITS + 1};
    SinewMind sm = sm_alloc(layers, array_length_u64_m(layers), sigmoid_f64, sigmoid_diff_f64);
    sm_random(&sm, &rng, 0.0, 1.0);
    sm_print(&sm, "Initial Configuration");
    printf("\nInitial Cost = %.5f\n", sm_cost(&sm, &ti, &to));
    printf("\n--------------------------------------------\n\n");

    f64 learning_rate = 1.0;
    f64 batch_size = ti.rows;
    for(u64 i = 0; i < 100 * 1000; ++i) {
        sm_backprop(&sm, &ti, &to);
        sm_step(&sm, learning_rate, batch_size);
    }

    sm_print(&sm, "Final Configuration");
    printf("\nFinal Cost = %.5f\n", sm_cost(&sm, &ti, &to));

    f64 compare_data[to.rows * to.cols];
    SM_Matrix compare = {
        .rows = to.rows, 
        .cols = to.cols, 
        .stride = to.cols, 
        .data = compare_data, 
        .parent = NULL
    };

    for(u64 s = 0; s < ti.rows; ++s) {
        SM_Matrix x = matrix_row_view_m(ti, s, 1);

        matrix_copy_m(sm.activations[0], x);
        sm_forward(&sm);

        for(u64 o = 0; o < to.cols; ++o) {
            matrix_pos_m(compare, s, o) = matrix_pos_m(sm.activations[sm.count], s, o);
        }
    }
    matrix_print(&to, "Expected Output", 0);
    matrix_print(&compare, "Actual Output", 0);

    sm_free(&sm);

    return 0;
}