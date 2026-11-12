#ifndef SM_RNG_H
#define SM_RNG_H

#include "sm_utils.h"
#include "sm_platform.h"
#include <time.h>

typedef struct SM_Rng {
    u64 states[4];
} SM_Rng;

static inline u64 rng_splitmix64(u64 *seed) {
    u64 num = (*seed += 0x9E3779B97F4A7C15ull);
    num = (num ^ (num >> 30)) * 0xBF58476D1CE4E5B9ull;
    num = (num ^ (num >> 27)) * 0x94D049BB133111EBull;
    return num ^ (num >> 31);
}

static inline u64 rng_hash_cstr_(const char *str) {
    u64 hash = 0xCBF29CE484222325ull;
    while(*str) {
        hash ^= (u64)(uchar)*str++;
        hash *= 0x100000001B3ull;
    }
    return hash;
}

static inline u64 rng_seed(SM_Rng *rng) {
    u64 seed = 0;

    seed ^= (u64)time(NULL);
    seed ^= (u64)clock() << 20;
    seed ^= (u64)(uptr)&seed;
    seed ^= sm_getpid_m() << 40;
    seed ^= rng_hash_cstr_(__DATE__ " " __TIME__);

    u64 seed_mix = seed;
    for(int i = 0; i < 4; ++i) {
        rng->states[i] = rng_splitmix64(&seed_mix);
    }

    return seed;
}

static inline void rng_seed_with(SM_Rng *rng, u64 seed) {
    u64 seed_mix = seed;
    for(int i = 0; i < 4; ++i) {
        rng->states[i] = rng_splitmix64(&seed_mix);
    }
}

// This is xoshiro256**
static inline u64 rng_next(SM_Rng *rng) {
    const u64 result = rotate_left_u64(rng->states[1] * 5, 7) * 9;
    const u64 saved = rng->states[1] << 17;
    
    rng->states[2] ^= rng->states[0];
    rng->states[3] ^= rng->states[1];
    rng->states[1] ^= rng->states[2];
    rng->states[0] ^= rng->states[3];
    rng->states[2] ^= saved;
    
    rng->states[3] = rotate_left_u64(rng->states[3], 45);
    return result;
}

static inline f64 rng_f64(SM_Rng *rng) {
    return (rng_next(rng) >> 11) * 0x1.0p-53;
}

#endif