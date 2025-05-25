#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spmv_test) {

    // Initialize operands
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 5;
    int *A_col = new int[A_nnz]{0, 1, 1, 0, 2};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 2, 3, 5};
    double *A_val = new double[A_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    double *x = new double[A_n_cols];
    for (int i = 0; i < A_n_cols; ++i) {
        x[i] = 1.0;
    }

    // Initialize result
    double *y = new double[A_n_rows];

    // Initialize expected result
    double *y_expected = new double[A_n_rows]{2.3, 2.2, 6.4};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("my_spmv", SMAX::KernelType::SPMV);
    smax->kernel("my_spmv")->register_A(A_n_rows, A_n_cols, A_nnz, A_col,
                                        A_row_ptr, A_val);
    smax->kernel("my_spmv")->register_B(A_n_cols, x);
    smax->kernel("my_spmv")->register_C(A_n_rows, y);

    // Function to test
    smax->kernel("my_spmv")->run();

    compare_arrays(y, y_expected, A_n_rows, "spmv_y");

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] x;
    delete[] y;
    delete[] y_expected;
    delete smax;
}