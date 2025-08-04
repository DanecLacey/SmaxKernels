#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spmv_test) {

    // Initialize operands
    CRSMatrix<int, double> *A = new CRSMatrix<int, double>(3, 3, 5);
    A->col = new int[A->nnz]{0, 1, 1, 0, 2};
    A->row_ptr = new int[A->n_rows + 1]{0, 2, 3, 5};
    A->val = new double[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    double *x = new double[A->n_cols];
    for (int i = 0; i < A->n_cols; ++i) {
        x[i] = 1.0;
    }

    // Initialize result
    double *y = new double[A->n_rows];

    // Initialize expected result
    double *y_expected = new double[A->n_rows]{2.3, 2.2, 6.4};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("my_spmv", SMAX::KernelType::SPMV);
    smax->kernel("my_spmv")->register_A(A->n_rows, A->n_cols, A->nnz, A->col,
                                        A->row_ptr, A->val);
    smax->kernel("my_spmv")->register_B(A->n_cols, x);
    smax->kernel("my_spmv")->register_C(A->n_rows, y);

    // Function to test
    smax->kernel("my_spmv")->run();

    compare_arrays(y, y_expected, A->n_rows, "spmv_y");

    delete A;
    delete[] x;
    delete[] y;
    delete[] y_expected;
    delete smax;
}