#include "sinew_mind.h"
#include <stdio.h>

f64 xor_train[] = {
    0.0, 0.0, 0.0,
    0.0, 1.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 1.0, 0.0
};

int main(void) {
    SM_Rng rng;
    rng_seed(&rng);

    SM_Matrix td = {
        .rows = 4,
        .cols = 3,
        .stride = 3,
        .data = xor_train,
        .parent = NULL,
    };
    SM_Matrix ti = matrix_col_view_m(td, 0, 2);
    SM_Matrix to = matrix_col_view_m(td, 2, 1);
    printf("\n--------------------------------------------\n\n");
    matrix_print(&ti, "Input", 0);
    matrix_print(&to, "Expected Output", 0);
    printf("\n--------------------------------------------\n\n");

    u64 layers[] = {2, 2, 1};
    SinewMind sm = sm_alloc(layers, array_length_u64_m(layers), sigmoid_f64);
    sm_random(&sm, &rng, 0.0, 1.0);
    sm_print(&sm, "Initial Configuration");
    printf("\nInitial Cost = %.5f\n", sm_cost(&sm, &ti, &to));
    printf("\n--------------------------------------------\n\n");

    

    return 0;
}