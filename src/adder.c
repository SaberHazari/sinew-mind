#include "core/sinew_mind.h"
#include <stdlib.h>
#include <stdio.h>

#define BITS 6

int main(int argc, char **argv) {
    SM_Rng rng;
    u64 seed = argc > 1 ? strtoull(argv[1], NULL, 10) : rng_seed(&rng);
    rng_seed_with(&rng, seed);
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
    SM_DataSplit split = sm_train_test_split(&ti, &to, &rng, 0.2);

    u64 layers[] = {2 * BITS, 2 * BITS + 1, BITS + 1};
    SinewMind sm = sm_alloc(layers, array_length_u64_m(layers), sm_activate_sigmoid, 
        sm_derivative_sigmoid);
    sm_random(&sm, &rng, 0.0, 1.0);
    printf("\nInitial Training Cost = %.10f\n", sm_cost(&sm, &split.x_train, &split.y_train));
    printf("\n--------------------------------------------\n\n");

    u64 epochs = 15 * 1000;
    u64 batch_size = split.x_train.rows / 10;
    f64 learning_rate = 5.0 * (f64)batch_size / (f64)split.x_train.rows;
    sm_train(&sm, &split.x_train, &split.y_train, &rng, epochs, batch_size, learning_rate);

    printf("\nFinal Training Cost = %.10f\n", sm_cost(&sm, &split.x_train, &split.y_train));

    SM_Metrics metrics = sm_test(&sm, &split.x_test, &split.y_test, 0.5, true);
    sm_metrics_print(&metrics);

    sm_data_split_free(&split);
    sm_free(&sm);

    return 0;
}
