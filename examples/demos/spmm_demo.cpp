/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-multiple vector multiplication (SpMM).
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

#define N_VECTORS 4

int main(void) {
    // Initialize operands
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 5;
    int *A_col = new int[A_nnz]{0, 1, 1, 0, 2};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 2, 3, 5};
    double *A_val = new double[A_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    double *X = new double[A_n_cols * N_VECTORS];
    for (int i = 0; i < A_n_cols * N_VECTORS; ++i) {
        X[i] = 1.0;
    }

    // Initialize result
    double *Y = new double[A_n_rows * N_VECTORS];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("useful_spmm", SMAX::KernelType::SPMM);

    // Register operands to this kernel tag
    // A is assumed to be in CRS format
    smax->kernel("useful_spmm")
        ->register_A(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);
    // X and Y are dense matrices
    smax->kernel("useful_spmm")->register_B(A_n_cols, N_VECTORS, X);
    smax->kernel("useful_spmm")->register_C(A_n_rows, N_VECTORS, Y);

    // Optionally, tell SMAX how to access dense block vector
    smax->kernel("useful_spmm")->set_vec_row_major(true);

    // Execute all phases of this kernel
    smax->kernel("useful_spmm")->run();

    smax->utils->print_timers();

    print_vector<double>(Y, A_n_cols * N_VECTORS);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] X;
    delete[] Y;
    delete smax;

    return 0;
}