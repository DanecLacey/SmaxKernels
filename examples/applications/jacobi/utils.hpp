#pragma once

#include "../../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include <iostream>

void peel_diag_crs(CRSMatrix *A, DenseMatrix *D) {

    for (int row_idx = 0; row_idx < A->n_rows; ++row_idx) {
        int row_start = A->row_ptr[row_idx];
        int row_end = A->row_ptr[row_idx + 1] - 1;
        int diag_j = -1; // Init diag col

        // find the diag in this row_idx (since row need not be col sorted)
        for (int j = row_start; j <= row_end; ++j) {
            if (A->col[j] == row_idx) {
                diag_j = j;
                D->val[row_idx] = A->val[j]; // extract
                if (std::abs(D->val[row_idx]) < 1e-16) {
                    printf("Zero diag!\n");
                    exit(EXIT_FAILURE);
                }
            }
        }
        if (diag_j < 0) {
            printf("No diag!\n");
            exit(EXIT_FAILURE);
        }

        // if it's not already at the end, swap it into the last slot
        if (diag_j != row_end) {
            std::swap(A->col[diag_j], A->col[row_end]);
            std::swap(A->val[diag_j], A->val[row_end]);
        }
    };
}

void normalize_x(DenseMatrix *x_new, DenseMatrix *x_old, DenseMatrix *D,
                 DenseMatrix *b) {

    int n_rows = b->n_rows;

    // #pragma omp parallel for
    for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
        double diag_contrib = D->val[row_idx] * x_old->val[row_idx];
        double offdiag_sum = x_new->val[row_idx] - diag_contrib;

        x_new->val[row_idx] = (b->val[row_idx] - offdiag_sum) / D->val[row_idx];
    }
}
