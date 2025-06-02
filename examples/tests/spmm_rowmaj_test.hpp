#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

#define N_VECTORS 4

REGISTER_TEST(spmm_rowmaj_test) {

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

    // Initialize expected result
    double *Y_expected = new double[A_n_rows * N_VECTORS]{
        2.3, 2.3, 2.3, 2.3, 2.2, 2.2, 2.2, 2.2, 6.4, 6.4, 6.4, 6.4};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("spmm", SMAX::KernelType::SPMM);
    smax->kernel("spmm")->register_A(A_n_rows, A_n_cols, A_nnz, A_col,
                                     A_row_ptr, A_val);
    smax->kernel("spmm")->register_B(A_n_cols, N_VECTORS, X);
    smax->kernel("spmm")->register_C(A_n_rows, N_VECTORS, Y);

    smax->kernel("spmm")->set_vec_row_major(true);

    // Function to test
    smax->kernel("spmm")->run();

    compare_arrays(Y, Y_expected, A_n_rows * N_VECTORS, "spmm_Y");

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] X;
    delete[] Y;
    delete[] Y_expected;
    delete smax;
}