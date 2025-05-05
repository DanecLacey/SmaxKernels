#include "../../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include <iostream>

CRSMatrix *create1DPoissonMatrixCRS(int n) {
    int N = n - 2;       // internal nodes (excluding Dirichlet boundaries)
    int nnz = 3 * N - 2; // maximum possible non-zero elements

    CRSMatrix *A = new CRSMatrix(N, N, nnz); // Square matrix, since Poisson

    int values_idx = 0;

    for (int i = 0; i < N; ++i) {
        // Diagonal entry
        A->values[values_idx] = 2.0;
        A->col[values_idx] = i;
        ++values_idx;

        // Left neighbor (if not first row)
        if (i > 0) {
            A->values[values_idx] = -1.0;
            A->col[values_idx] = i - 1;
            ++values_idx;
        }

        // Right neighbor (if not last row)
        if (i < N - 1) {
            A->values[values_idx] = -1.0;
            A->col[values_idx] = i + 1;
            ++values_idx;
        }

        // Update row_ptr to point to the next row's start index in values
        A->row_ptr[i + 1] = values_idx;
    }

    // Set the number of rows, columns, and non-zero elements
    A->n_rows = N;
    A->n_cols = N;       // Assume the matrix is square
    A->nnz = values_idx; // The total number of non-zero entries

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