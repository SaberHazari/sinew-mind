#include "sinew_mind.h"

int main(void) {
    u64 layers[] = {28 * 28, 16, 16, 10};
    SinewMind sm = sm_alloc(layers, array_length_u64_m(layers));
    sm_print_with(&sm, "Sinew Mind");

    return 0;
}