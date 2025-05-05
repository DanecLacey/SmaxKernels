#include "../../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include <iostream>

CRSMatrix create1DPoissonMatrixCRS(int n) {
    int N = n - 2;       // internal nodes (excluding Dirichlet boundaries)
    int nnz = 3 * N - 2; // maximum possible non-zero elements

    CRSMatrix A(N, N, nnz); // Square matrix, since Poisson

    int values_idx = 0;

    for (int i = 0; i < N; ++i) {
        // Diagonal entry
        A.values[values_idx] = 2.0;
        A.col[values_idx] = i;
        ++values_idx;

        // Left neighbor (if not first row)
        if (i > 0) {
            A.values[values_idx] = -1.0;
            A.col[values_idx] = i - 1;
            ++values_idx;
        }

        // Right neighbor (if not last row)
        if (i < N - 1) {
            A.values[values_idx] = -1.0;
            A.col[values_idx] = i + 1;
            ++values_idx;
        }

        // Update row_ptr to point to the next row's start index in values
        A.row_ptr[i + 1] = values_idx;
    }

    // Set the number of rows, columns, and non-zero elements
    A.n_rows = N;
    A.n_cols = N;       // Assume the matrix is square
    A.nnz = values_idx; // The total number of non-zero entries

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

double infty_vec_norm(double *values, int n_rows) {
    double max_abs = 0.0;
    double curr_abs;
    for (int i = 0; i < n_rows; ++i) {
        curr_abs = (values[i] >= 0) ? values[i] : -1 * values[i];
        if (curr_abs > max_abs) {
            max_abs = curr_abs;
        }
    }

    return max_abs;
};

void extract_D_L_U(const CRSMatrix &A, CRSMatrix &D_plus_L, CRSMatrix &U) {
    // Count nnz
    for (int i = 0; i < A.n_rows; ++i) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A.col[idx];

            if (col <= i) {
                ++D_plus_L.nnz;
            } else {
                ++U.nnz;
            }
        }
    }

    // Allocate heap space and assign known metadata
    D_plus_L.values = new double[D_plus_L.nnz];
    D_plus_L.col = new int[D_plus_L.nnz];
    D_plus_L.row_ptr = new int[A.n_rows + 1];
    D_plus_L.row_ptr[0] = 0;
    D_plus_L.n_rows = A.n_rows;
    D_plus_L.n_cols = A.n_cols;

    U.values = new double[U.nnz];
    U.col = new int[U.nnz];
    U.row_ptr = new int[A.n_rows + 1];
    U.row_ptr[0] = 0;
    U.n_rows = A.n_rows;
    U.n_cols = A.n_cols;

    // Assign nonzeros
    int D_plus_L_count = 0;
    int U_count = 0;
    for (int i = 0; i < A.n_rows; ++i) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A.col[idx];
            double val = A.values[idx];

            if (col <= i) {
                // Diagonal or lower triangular part (D + L)
                D_plus_L.values[D_plus_L_count] = val;
                D_plus_L.col[D_plus_L_count++] = col;
            } else {
                // Strictly upper triangular part (U)
                U.values[U_count] = val;
                U.col[U_count++] = col;
            }
        }

        // Update row pointers
        D_plus_L.row_ptr[i + 1] = D_plus_L_count;
        U.row_ptr[i + 1] = U_count;
    }
}

#define REGISTER_SPMV_KERNEL(kernel_name, mat, X, Y)                           \
    smax->register_kernel(kernel_name, SMAX::SPMV, SMAX::CPU);                 \
    smax->kernels[kernel_name]->register_A(                                    \
        mat.n_rows, mat.n_cols, mat.nnz, &mat.col, &mat.row_ptr, &mat.values); \
    smax->kernels[kernel_name]->register_B(mat.n_cols, &X->values);            \
    smax->kernels[kernel_name]->register_C(mat.n_rows, &Y->values);

#define REGISTER_SPTRSV_KERNEL(kernel_name, mat, X, Y)                         \
    smax->register_kernel(kernel_name, SMAX::SPTRSV, SMAX::CPU);               \
    smax->kernels[kernel_name]->register_A(                                    \
        mat.n_rows, mat.n_cols, mat.nnz, &mat.col, &mat.row_ptr, &mat.values); \
    smax->kernels[kernel_name]->register_B(mat.n_cols, &X->values);            \
    smax->kernels[kernel_name]->register_C(mat.n_rows, &Y->values);
