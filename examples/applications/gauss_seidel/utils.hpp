#ifndef SMAX_GAUSS_SEIDEL_UTILS
#define SMAX_GAUSS_SEIDEL_UTILS

#include "../../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include <iostream>

CRSMatrix *create1DPoissonMatrixCRS(int n) {
    int N = n - 2;       // internal nodes (excluding Dirichlet boundaries)
    int nnz = 3 * N - 2; // maximum possible non-zero elements

    CRSMatrix *A = new CRSMatrix(N, N, nnz); // Square matrix, since Poisson

    int val_idx = 0;

    for (int i = 0; i < N; ++i) {
        // Diagonal entry
        A->val[val_idx] = 2.0;
        A->col[val_idx] = i;
        ++val_idx;

        // Left neighbor (if not first row)
        if (i > 0) {
            A->val[val_idx] = -1.0;
            A->col[val_idx] = i - 1;
            ++val_idx;
        }

        // Right neighbor (if not last row)
        if (i < N - 1) {
            A->val[val_idx] = -1.0;
            A->col[val_idx] = i + 1;
            ++val_idx;
        }

        // Update row_ptr to point to the next row's start index in val
        A->row_ptr[i + 1] = val_idx;
    }

    // Set the number of rows, columns, and non-zero elements
    A->n_rows = N;
    A->n_cols = N;    // Assume the matrix is square
    A->nnz = val_idx; // The total number of non-zero entries

    return A;
}

void subtract_vectors(double *result_vec, const double *vec1,
                      const double *vec2, const int N,
                      const double scale = 1.0) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        result_vec[i] = vec1[i] - scale * vec2[i];
    }
};

double infty_vec_norm(double *val, int n_rows) {
    double max_abs = 0.0;
    double curr_abs;
    for (int i = 0; i < n_rows; ++i) {
        curr_abs = (val[i] >= 0) ? val[i] : -1 * val[i];
        if (curr_abs > max_abs) {
            max_abs = curr_abs;
        }
    }

    return max_abs;
};

inline void naive_crs_sptrsv(int A_n_rows, int A_n_cols, int A_nnz,
                             int *RESTRICT A_col, int *RESTRICT A_row_ptr,
                             double *RESTRICT A_val, double *RESTRICT X,
                             double *RESTRICT Y) {
    for (int i = 0; i < A_n_rows; ++i) {
        double sum = 0.0;
        double diag = 0.0;

        for (int j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {

            double val = A_val[j];

            if (A_col[j] < i) {
                sum += val * X[A_col[j]];
            } else if (A_col[j] == i) {
                diag = val;
                // printf("diag = %f\n", diag);
            } else {
                printf("row: %d, col: %d, val: %f\n", i, A_col[j], val);
            }
        }

        X[i] = (Y[i] - sum) / diag;
    }
}

#endif // SMAX_GAUSS_SEIDEL_UTILS
