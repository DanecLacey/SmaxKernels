#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(sptrsv_test) {

    // Initialize operands
    CRSMatrix<int, double> *A = new CRSMatrix<int, double>(3, 3, 4);
    A->col = new int[A->nnz]{0, 1, 0, 2};
    A->row_ptr = new int[A->n_rows + 1]{0, 1, 2, 4};
    A->val = new double[A->nnz]{1.1, 2.2, 3.1, 3.3};

    double *x = new double[A->n_cols];
    for (int i = 0; i < A->n_cols; ++i) {
        x[i] = 1.0;
    }

    // Initialize RHS
    double *b = new double[A->n_cols];
    for (int i = 0; i < A->n_cols; ++i) {
        b[i] = 2.0;
    }

    // Initialize expected result
    double *x_expected = new double[A->n_rows]{
        1.81818181818181812, 0.90909090909090906, -1.10192837465564764};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("sptrsv", SMAX::KernelType::SPTRSV);
    smax->kernel("sptrsv")->register_A(A->n_rows, A->n_cols, A->nnz, A->col,
                                       A->row_ptr, A->val);
    smax->kernel("sptrsv")->register_B(A->n_cols, x);
    smax->kernel("sptrsv")->register_C(A->n_rows, b);

    // Function to test
    smax->kernel("sptrsv")->run();

    compare_arrays(x, x_expected, A->n_rows, "sptrsv_x");

    delete A;
    delete[] x;
    delete[] b;
    delete[] x_expected;
    delete smax;
}