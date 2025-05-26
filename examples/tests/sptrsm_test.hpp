#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

#define N_VECTORS 4

REGISTER_TEST(sptrsm_test) {

    // Initialize operands
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 4;
    int *A_col = new int[A_nnz]{0, 1, 0, 2};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 1, 2, 4};
    double *A_val = new double[A_nnz]{1.1, 2.2, 3.1, 3.3};

    double *X = new double[A_n_cols * N_VECTORS];
    for (int i = 0; i < A_n_cols * N_VECTORS; ++i) {
        X[i] = 1.0;
    }

    // Initialize RHS
    double *B = new double[A_n_cols * N_VECTORS];
    for (int i = 0; i < A_n_cols * N_VECTORS; ++i) {
        B[i] = 2.0;
    }

    // Initialize expected result
    double *X_expected = new double[A_n_rows * N_VECTORS]{
        1.81818181818181812, 0.90909090909090906, -1.10192837465564764,
        1.81818181818181812, 0.90909090909090906, -1.10192837465564764,
        1.81818181818181812, 0.90909090909090906, -1.10192837465564764,
        1.81818181818181812, 0.90909090909090906, -1.10192837465564764};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("sptrsm", SMAX::KernelType::SPTRSM);
    smax->kernel("sptrsm")->register_A(A_n_rows, A_n_cols, A_nnz, A_col,
                                       A_row_ptr, A_val);
    smax->kernel("sptrsm")->register_B(A_n_cols, N_VECTORS, X);
    smax->kernel("sptrsm")->register_C(A_n_rows, N_VECTORS, B);

    // Function to test
    smax->kernel("sptrsm")->run();

    compare_arrays(X, X_expected, A_n_rows * N_VECTORS, "sptrsm_X");

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] X;
    delete[] B;
    delete[] X_expected;
    delete smax;
}