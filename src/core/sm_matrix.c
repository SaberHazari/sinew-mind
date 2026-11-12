#include "sm_matrix.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

static bool matrix_overlaps_(const SM_Matrix *x, const SM_Matrix *y) {
    usize x_length = x->rows ? ((x->rows - 1) * x->stride + x->cols) : 0;
    usize y_length = y->rows ? ((y->rows - 1) * y->stride + y->cols) : 0;
    return ranges_overlap(x->data, x_length * sizeof(*x->data), 
        y->data, y_length * sizeof(*y->data));
}

SM_Matrix matrix_alloc(u64 rows, u64 cols) {
    assert(rows && cols && cols <= UINT64_MAX / rows);

    SM_Matrix mat;
    mat.parent = NULL;
    mat.rows = rows;
    mat.cols = cols;
    mat.stride = cols;
    // I'm using malloc temporarily
    mat.data = malloc(rows * cols * sizeof(*mat.data));
    assert(mat.data);
    
    return mat;
}

void matrix_random(SM_Matrix *mat, SM_Rng *rng, f64 lo, f64 hi) {
    assert(lo < hi);

    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            matrix_pos_ptr_m(mat, i, j) = lo + rng_f64(rng) * (hi - lo);
        }
    }
}

SM_Matrix matrix_subview(SM_Matrix *mat, u64 row_start, u64 row_count, 
    u64 col_start, u64 col_count) {
    assert(row_start <= mat->rows && row_count <= mat->rows - row_start);
    assert(col_start <= mat->cols && col_count <= mat->cols - col_start);

    SM_Matrix sub;
    sub.parent = mat;
    sub.rows = row_count;
    sub.cols = col_count;
    sub.stride = mat->stride;
    sub.data = mat->data + row_start * mat->stride + col_start;

    return sub;
}

void matrix_submat(SM_Matrix *sub, SM_Matrix *mat, 
    u64 row_start, u64 row_count, 
    u64 col_start, u64 col_count) {
    assert(row_start <= mat->rows && row_count <= mat->rows - row_start);
    assert(col_start <= mat->cols && col_count <= mat->cols - col_start);
    assert(sub->rows == row_count && sub->cols == col_count);

    for(u64 i = 0; i < sub->rows; ++i) {
        for(u64 j = 0; j < sub->cols; ++j) {
            matrix_pos_ptr_m(sub, i, j) = matrix_pos_ptr_m(mat, row_start + i, col_start + j);
        }
    }
}

void matrix_add(SM_Matrix *dest, SM_Matrix *x) {
    assert(dest->rows == x->rows && dest->cols == x->cols);

    for(u64 i = 0; i < dest->rows; ++i) {
        for(u64 j = 0; j < dest->cols; ++j) {
            matrix_pos_ptr_m(dest, i, j) += matrix_pos_ptr_m(x, i, j);
        }
    }
}

void matrix_mul(SM_Matrix *dest, SM_Matrix *x, SM_Matrix *y) {
    assert(x->cols == y->rows && dest->rows == x->rows && 
        dest->cols == y->cols);
    assert(!matrix_overlaps_(dest, x));
    assert(!matrix_overlaps_(dest, y));
    
    for(u64 i = 0; i < dest->rows; ++i) {
        f64 *dest_i_row = &dest->data[i * dest->stride];
        for(u64 j = 0; j < dest->cols; ++j) { dest_i_row[j] = 0; }
        for(u64 k = 0; k < x->cols; ++k) {
            f64 x_ik = matrix_pos_ptr_m(x, i, k);
            f64 *y_k_row = &y->data[k * y->stride];
            for(u64 j = 0; j < dest->cols; ++j) {
                dest_i_row[j] += x_ik * y_k_row[j];
            }
        }
    }
}

void matrix_activate(SM_Matrix *mat, ActivationFn activator) {
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            matrix_pos_ptr_m(mat, i, j) = activator(matrix_pos_ptr_m(mat, i, j));
        }
    }
}

void matrix_print(SM_Matrix *mat, const char *name, u32 padding) {
    printf("%*s%s = [\n", padding, "", name);
    for(u64 i = 0; i < mat->rows; ++i) {
        printf("%*s", padding, "");
        for(u64 j = 0; j < mat->cols; ++j) {
            printf("  %.5f", matrix_pos_ptr_m(mat, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", padding, "");
}

void matrix_free(SM_Matrix *mat) {
    if(mat->parent == NULL) {
        free(mat->data);
    }
    mat->data = NULL;
    mat->rows = mat->cols = mat->stride = 0;
    mat->parent = NULL;
}