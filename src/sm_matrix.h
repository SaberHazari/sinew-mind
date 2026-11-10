#ifndef SM_MATRIX_H
#define SM_MATRIX_H

#include "sm_utils.h"
#include "sm_rng.h"

#define matrix_pos_ptr_m(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])
#define matrix_rng_0_1_m(mat, rng) matrix_rng(&(mat), &(rng), 0.0, 1.0)
#define matrix_row_view_m(mat, row_start, row_count) matrix_subview(&(mat), (row_start), (row_count), 0, (mat).cols)
#define matrix_col_view_m(mat, col_start, col_count) matrix_subview(&(mat), 0, (mat).rows, (col_start), (col_count))
#define matrix_row_m(sub, mat, row_start, row_count) matrix_submat(&(sub), &(mat), (row_start), (row_count), 0, (mat).cols)
#define matrix_col_m(sub, mat, col_start, col_count) matrix_submat(&(sub), &(mat), 0, (mat).rows, (col_start), (col_count))
#define matrix_copy_m(dest, src) matrix_submat(&(dest), &(src), 0, (src).rows, 0, (src).cols)
#define matrix_print_m(mat) matrix_print_with(&(mat), #mat)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SM_Matrix SM_Matrix;
struct SM_Matrix {
    u64 rows;
    u64 cols;
    u64 stride;
    f64 *data;
    SM_Matrix *parent;
};

SM_Matrix matrix_alloc(u64 rows, u64 cols);
void matrix_rng(SM_Matrix *mat, SM_Rng *rng, f64 lo, f64 hi);
SM_Matrix matrix_subview(SM_Matrix *mat, u64 row_start, u64 row_count, 
    u64 col_start, u64 col_count);
void matrix_submat(SM_Matrix *sub, SM_Matrix *mat, u64 row_start, u64 row_count, 
    u64 col_start, u64 col_count);
void matrix_add(SM_Matrix *dest, SM_Matrix *x);
void matrix_mul(SM_Matrix *dest, SM_Matrix *x, SM_Matrix *y);
void matrix_sigmoid(SM_Matrix *mat);
void matrix_print_with(SM_Matrix *mat, const char *name);
// This free is temporary until I add a memory arena
void matrix_free(SM_Matrix *mat);

#ifdef __cplusplus
}
#endif

#endif