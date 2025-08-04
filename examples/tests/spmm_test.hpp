#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spmm_test) {

    int n_vectors = 4;

    // Initialize operands
    CRSMatrix<int, double> *A = new CRSMatrix<int, double>(3, 3, 5);
    A->col = new int[A->nnz]{0, 1, 1, 0, 2};
    A->row_ptr = new int[A->n_rows + 1]{0, 2, 3, 5};
    A->val = new double[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    double *X = new double[A->n_cols * n_vectors];
    for (int i = 0; i < A->n_cols * n_vectors; ++i) {
        X[i] = 1.0;
    }

    // Initialize result
    double *Y = new double[A->n_rows * n_vectors];

    // Initialize expected result
    double *Y_expected = new double[A->n_rows * n_vectors]{
        2.3, 2.2, 6.4, 2.3, 2.2, 6.4, 2.3, 2.2, 6.4, 2.3, 2.2, 6.4};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("spmm", SMAX::KernelType::SPMM);
    smax->kernel("spmm")->register_A(A->n_rows, A->n_cols, A->nnz, A->col,
                                     A->row_ptr, A->val);
    smax->kernel("spmm")->register_B(A->n_cols, n_vectors, X);
    smax->kernel("spmm")->register_C(A->n_rows, n_vectors, Y);

    // Function to test
    smax->kernel("spmm")->run();

    compare_arrays(Y, Y_expected, A->n_rows * n_vectors, "spmm_Y");

    delete A;
    delete[] X;
    delete[] Y;
    delete[] Y_expected;
    delete smax;
}