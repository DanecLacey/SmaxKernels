#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(sptrsm_rowmaj_test) {

    int n_vectors = 4;

    // Initialize operands
    CRSMatrix<int, double> *A = new CRSMatrix<int, double>(3, 3, 4);
    A->col = new int[A->nnz]{0, 1, 0, 2};
    A->row_ptr = new int[A->n_rows + 1]{0, 1, 2, 4};
    A->val = new double[A->nnz]{1.1, 2.2, 3.1, 3.3};

    double *X = new double[A->n_cols * n_vectors];
    for (int i = 0; i < A->n_cols * n_vectors; ++i) {
        X[i] = 1.0;
    }

    // Initialize RHS
    double *B = new double[A->n_cols * n_vectors];
    for (int i = 0; i < A->n_cols * n_vectors; ++i) {
        B[i] = 2.0;
    }

    // Initialize expected result
    double *X_expected = new double[A->n_rows * n_vectors]{
        1.81818181818181812,  1.81818181818181812,  1.81818181818181812,
        1.81818181818181812,  0.90909090909090906,  0.90909090909090906,
        0.90909090909090906,  0.90909090909090906,  -1.10192837465564764,
        -1.10192837465564764, -1.10192837465564764, -1.10192837465564764};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("sptrsm", SMAX::KernelType::SPTRSM);
    smax->kernel("sptrsm")->register_A(A->n_rows, A->n_cols, A->nnz, A->col,
                                       A->row_ptr, A->val);
    smax->kernel("sptrsm")->register_B(A->n_cols, n_vectors, X);
    smax->kernel("sptrsm")->register_C(A->n_rows, n_vectors, B);

    smax->kernel("sptrsm")->set_vec_row_major(true);

    // Function to test
    smax->kernel("sptrsm")->run();

    compare_arrays(X, X_expected, A->n_rows * n_vectors, "sptrsm_X");

    delete A;
    delete[] X;
    delete[] B;
    delete[] X_expected;
    delete smax;
}