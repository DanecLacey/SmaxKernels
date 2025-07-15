
#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(bspmv_laplace_test) {

    constexpr int mat_size = 1608;
    // constexpr int mat_size = 8;

    int A_crs_n_rows = mat_size;
    int A_crs_n_cols = mat_size;
    int A_crs_nnz = 3*mat_size -2;
    int *A_crs_col = new int[A_crs_nnz];
    int *A_crs_row_ptr = new int[A_crs_n_rows + 1];
    double *A_crs_val = new double[A_crs_nnz];

    A_crs_col[0] = int(0);
    A_crs_col[1] = int(1);
    A_crs_row_ptr[0] = int(0);
    A_crs_row_ptr[1] = int(2);
    A_crs_val[0] = double(2);
    A_crs_val[1] = double(-1);
    for(int i = 1; i < A_crs_n_rows-1; ++i)
    {
      A_crs_col[2+(i-1)*3] = int(i-1);
      A_crs_col[2+(i-1)*3+1] = int(i);
      A_crs_col[2+(i-1)*3+2] = int(i+1);
      A_crs_row_ptr[i+1] = A_crs_row_ptr[i] + int(3);
      A_crs_val[2+(i-1)*3] = double(-1);
      A_crs_val[2+(i-1)*3+1] = double(2);
      A_crs_val[2+(i-1)*3+2] = double(-1);
    }
    A_crs_col[A_crs_nnz-2] = int(A_crs_n_rows-2);
    A_crs_col[A_crs_nnz-1] = int(A_crs_n_rows-1);
    A_crs_row_ptr[A_crs_n_rows] = int(A_crs_nnz);
    A_crs_val[A_crs_nnz-2] = double(-1);
    A_crs_val[A_crs_nnz-1] = double(2);

    const bool use_blocked_column_major = false;
    const int target_b_height = 4;
    const int target_b_width = 8;
    const int target_height_pad = 4;
    const int target_width_pad = 8;

    // Declare bcrs operand
    int A_bcrs_n_rows = 0;
    int A_bcrs_n_cols = 0;
    int A_bcrs_b_width = 0;
    int A_bcrs_b_height = 0;
    int A_bcrs_b_w_pad = 0;
    int A_bcrs_b_h_pad = 0;
    int A_bcrs_nnz = 0;
    int *A_bcrs_col = nullptr;
    int *A_bcrs_row_ptr = nullptr;
    double *A_bcrs_val = nullptr;

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_bcrs<int, double, int>(
        A_crs_n_rows, A_crs_n_cols, A_crs_nnz, A_crs_col, A_crs_row_ptr,
        A_crs_val, A_bcrs_n_rows, A_bcrs_n_cols, A_bcrs_nnz, A_bcrs_b_height, A_bcrs_b_width, A_bcrs_b_h_pad, A_bcrs_b_w_pad,
        A_bcrs_col, A_bcrs_row_ptr, A_bcrs_val, target_b_height, target_b_width, target_height_pad, target_width_pad, use_blocked_column_major);

    // std::vector<std::string> stri

    double *x = new double[A_bcrs_n_cols*A_bcrs_b_w_pad];
    for (int i = 0; i < A_bcrs_n_cols*A_bcrs_b_width; ++i) {
        x[i] = 1.0/double((i+3)%23+1);
    }

    // Initialize result
    double *y = new double[A_bcrs_n_rows * A_bcrs_b_h_pad];

    // Initialize expected result
    double *y_expected = new double[A_bcrs_n_rows * A_bcrs_b_h_pad];

    smax->register_kernel("my_spmv", SMAX::KernelType::SPMV);
    smax->kernel("my_spmv")->register_A(A_crs_n_rows, A_crs_n_cols, A_crs_nnz, A_crs_col,
                                        A_crs_row_ptr, A_crs_val);
    smax->kernel("my_spmv")->register_B(A_crs_n_cols, x);
    smax->kernel("my_spmv")->register_C(A_crs_n_rows, y_expected);

    // Function to test
    smax->kernel("my_spmv")->run();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("my_bspmv", SMAX::KernelType::BSPMV);
    smax->kernel("my_bspmv")->register_A(A_bcrs_n_rows, A_bcrs_n_cols, A_bcrs_nnz,
                                        A_bcrs_b_height, A_bcrs_b_width, A_bcrs_b_h_pad,
                                        A_bcrs_b_w_pad, A_bcrs_col, A_bcrs_row_ptr, A_bcrs_val);
    smax->kernel("my_bspmv")->register_B(A_bcrs_n_cols*A_bcrs_b_w_pad, x);
    smax->kernel("my_bspmv")->register_C(A_bcrs_n_rows*A_bcrs_b_h_pad, y);
    smax->kernel("my_bspmv")->set_block_column_major(use_blocked_column_major);

    // Function to test
    smax->kernel("my_bspmv")->run();

    compare_arrays(y, y_expected, A_bcrs_n_rows * A_bcrs_b_h_pad, "spmv_y");

    delete[] A_crs_col;
    delete[] A_crs_row_ptr;
    delete[] A_crs_val;
    delete[] A_bcrs_col;
    delete[] A_bcrs_row_ptr;
    delete[] A_bcrs_val;
    delete[] x;
    delete[] y;
    delete[] y_expected;
    delete smax;
}