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

    u64 layers[] = {2 * BITS, 2 * BITS, BITS + 1};
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

    u64 success = 0, failure = 0;

    for(u64 x = 0; x < n; ++x) {
        for(u64 y = 0; y < n; ++y) {
            for(u64 j = 0; j < BITS; ++j) {
                matrix_pos_m(sm.activations[0], 0, j) = (x >> j) & 1;
                matrix_pos_m(sm.activations[0], 0, j + BITS) = (y >> j) & 1;
            }
            sm_forward(&sm);

            u64 out_z = 0;
            u64 exp_z = x + y;
            for(u64 j = 0; j < to.cols; ++j) {
                u64 out_bit = matrix_pos_m(sm.activations[sm.count], 0, j) > 0.5;
                out_z |= out_bit << j;
            }
            printf("%" PRIu64 " + %" PRIu64 " = %" PRIu64 ", Actual Output = %" PRIu64 "\n", x, y, exp_z, out_z);

            if(out_z == exp_z) { success++; }
            else { failure++; }
        }
    }

    printf("Accuracy = %.5f\n", (f64)success / (success + failure));

    sm_free(&sm);

    return 0;
}