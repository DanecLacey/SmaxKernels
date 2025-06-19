#pragma once

#include "../spmv_helpers.hpp"
#include "../sptrsv_helpers.hpp"

#define PRINT_PERM_VECTOR(A, perm)                                             \
    do {                                                                       \
        printf("Using permutation vector: [");                                 \
        for (unsigned long long int _i = 0; _i < (A)->n_rows; ++_i) {          \
            printf("%d%s", (perm)[_i], (_i < (A)->n_rows - 1) ? ", " : "");    \
        }                                                                      \
        printf("]\n");                                                         \
    } while (0)

#define PRINT_ITER(n_iters, residual_norm)                                     \
    do {                                                                       \
        printf("iter: %d, residual_norm = %f\n", (n_iters), (residual_norm));  \
    } while (0)

CRSMatrix<int, double> *create1DPoissonMatrixCRS(int n) {
    int N = n - 2;       // internal nodes (excluding Dirichlet boundaries)
    int nnz = 3 * N - 2; // maximum possible non-zero elements

    CRSMatrix<int, double> *A =
        new CRSMatrix<int, double>(N, N, nnz); // Square matrix, since Poisson

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

CRSMatrix<int, double> *create2DPoissonMatrixCRS(int n) {
    int N = n * n;       // total number of grid points
    int max_nnz = 5 * N; // 5-point stencil: up to 5 non-zeros per row

    CRSMatrix<int, double> *A = new CRSMatrix<int, double>(N, N, max_nnz);

    int val_idx = 0;
    A->row_ptr[0] = 0;

    for (int i = 0; i < n; ++i) {     // row in 2D grid
        for (int j = 0; j < n; ++j) { // column in 2D grid
            int row = i * n + j;

            // Left neighbor
            if (j > 0) {
                A->val[val_idx] = -1.0;
                A->col[val_idx++] = row - 1;
            }

            // Bottom neighbor
            if (i > 0) {
                A->val[val_idx] = -1.0;
                A->col[val_idx++] = row - n;
            }

            // Diagonal
            A->val[val_idx] = 4.0;
            A->col[val_idx++] = row;

            // Top neighbor
            if (i < n - 1) {
                A->val[val_idx] = -1.0;
                A->col[val_idx++] = row + n;
            }

            // Right neighbor
            if (j < n - 1) {
                A->val[val_idx] = -1.0;
                A->col[val_idx++] = row + 1;
            }

            A->row_ptr[row + 1] = val_idx;
        }
    }

    A->n_rows = N;
    A->n_cols = N;
    A->nnz = val_idx;

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
