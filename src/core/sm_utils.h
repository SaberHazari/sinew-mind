#ifndef SM_UTILS_H
#define SM_UTILS_H

#include <inttypes.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

typedef unsigned char uchar;

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int64_t i64;

typedef double f64;

typedef size_t usize;
typedef uintptr_t uptr;

#define array_length_u64_m(arr) (u64)(sizeof((arr)) / sizeof((arr)[0]))

static inline bool ranges_overlap(const void *x, usize x_length, 
    const void *y, usize y_length) {
    if(!x_length || !y_length) { return false; }
    const u8 *x_ptr = x, *y_ptr = y;
    return (x_ptr < y_ptr + y_length) && (y_ptr < x_ptr + x_length);
}

static inline u64 rotate_left_u64(u64 x, u32 k) {
    k &= 63;
    return k ? (x << k) | (x >> (64 - k)) : x;
}

static inline i64 choice_round_div_i64(i64 x, i64 y, bool ceiling) {
    assert(y != 0 && !(x == INT64_MIN && y == -1));

    i64 q = x / y;
    i64 r = x % y;
    if(r != 0 && ((r < 0) == (y < 0)) == ceiling) {
        q += ceiling ? 1 : -1;
    }

    return q;
}

#endif