#ifndef SM_UTILS_H
#define SM_UTILS_H

#include <stdint.h>
#include <math.h>

typedef unsigned char uchar;

typedef uint32_t u32;
typedef uint64_t u64;

typedef double f64;

typedef uintptr_t uptr;

static inline u64 rotate_left_u64(u64 x, u32 k) {
    k &= 63;
    return k ? (x << k) | (x >> (64 - k)) : x;
}

static inline f64 sigmoid_f64(f64 x) {
    return 1.0 / (1.0 + exp(-x));
}

#endif