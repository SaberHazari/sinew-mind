#include "sm_matrix.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

SM_Matrix matrix_alloc(u64 rows, u64 cols) {
    SM_Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.stride = cols;
    // I'm using malloc temporarily
    mat.data = malloc(rows * cols * sizeof(*mat.data));
    mat.parent = NULL;
    assert(mat.data);
    
    return mat;
}

void matrix_rng(SM_Matrix *mat, SM_Rng *rng, f64 lo, f64 hi) {
    assert(lo < hi);
    if(lo >= hi) { lo = nextafter(hi, lo); }

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
    sub.rows = row_count;
    sub.cols = col_count;
    sub.stride = mat->stride;
    sub.data = mat->data + row_start * mat->stride + col_start;
    sub.parent = mat;

    return sub;
}

void matrix_submat(SM_Matrix *sub, SM_Matrix *mat, u64 row_start, u64 row_count, 
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
    
    for(u64 i = 0; i < dest->rows; ++i) {
        for(u64 j = 0; j < dest->cols; ++j) {
            matrix_pos_ptr_m(dest, i, j) = 0;
            for(u64 k = 0; k < x->cols; ++k) {
                matrix_pos_ptr_m(dest, i, j) += matrix_pos_ptr_m(x, i, k) * 
                    matrix_pos_ptr_m(y, k, j);
            }
        }
    }
}

void matrix_sigmoid(SM_Matrix *mat) {
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            matrix_pos_ptr_m(mat, i, j) = sigmoid_f64(matrix_pos_ptr_m(mat, i, j));
        }
    }
}

void matrix_print_with(SM_Matrix *mat, const char *name) {
    printf("%s = [\n", name);
    for(u64 i = 0; i < mat->rows; ++i) {
        for(u64 j = 0; j < mat->cols; ++j) {
            printf("  %.5f", matrix_pos_ptr_m(mat, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void matrix_free(SM_Matrix *mat) {
    if(mat->parent == NULL) {
        free(mat->data);
    }
    mat->data = NULL;
    mat->rows = mat->cols = mat->stride = 0;
    mat->parent = NULL;
}