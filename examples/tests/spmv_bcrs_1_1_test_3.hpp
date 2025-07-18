#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(bspmv_test) {

    // Initialize operands
    // Declare expected output from conversion utility
    const bool block_column_major = true;
    int A_bcrs_n_rows = 2;
    int A_bcrs_n_cols = 2;
    int A_bcrs_b_width = 2;
    int A_bcrs_b_height = 2;
    int A_bcrs_b_w_pad = 2;
    int A_bcrs_b_h_pad = 2;
    int A_bcrs_nnz = 3;
    int *A_bcrs_col = new int[A_bcrs_nnz]{0, 1, 1};
    int *A_bcrs_row_ptr = new int[A_bcrs_n_rows + 1]{0, 2, 3};
    double *A_bcrs_val = new double[A_bcrs_nnz*A_bcrs_b_w_pad*A_bcrs_b_h_pad]{1.1, 0.0, 0.0, 2.2, 1.3, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 4.4};

    double *x = new double[A_bcrs_n_cols*A_bcrs_b_w_pad];
    for (int i = 0; i < A_bcrs_n_cols*A_bcrs_b_width; ++i) {
        x[i] = 1.0;
    }

    // Initialize result
    double *y = new double[A_bcrs_n_rows * A_bcrs_b_h_pad];

    // Initialize expected result
    double *y_expected = new double[A_bcrs_n_rows * A_bcrs_b_h_pad]{3.8, 2.2, 0., 4.4};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("my_spmv", SMAX::KernelType::BSPMV);
    smax->kernel("my_spmv")->register_A(A_bcrs_n_rows, A_bcrs_n_cols, A_bcrs_nnz,
                                        A_bcrs_b_height, A_bcrs_b_width, A_bcrs_b_h_pad,
                                        A_bcrs_b_w_pad, A_bcrs_col, A_bcrs_row_ptr, A_bcrs_val);
    smax->kernel("my_spmv")->register_B(A_bcrs_n_cols*A_bcrs_b_w_pad, x);
    smax->kernel("my_spmv")->register_C(A_bcrs_n_rows*A_bcrs_b_h_pad, y);
    smax->kernel("my_spmv")->set_block_column_major(block_column_major);

    // Function to test
    smax->kernel("my_spmv")->run();

    compare_arrays(y, y_expected, A_bcrs_n_rows * A_bcrs_b_h_pad, "spmv_y");

    // register cuda kernel and compare, if available
    smax->register_kernel("my_spmv_cuda", SMAX::KernelType::BSPMV, SMAX::PlatformType::CUDA);
    smax->kernel("my_spmv_cuda")->register_A(A_bcrs_n_rows, A_bcrs_n_cols, A_bcrs_nnz,
                                        A_bcrs_b_height, A_bcrs_b_width, A_bcrs_b_h_pad,
                                        A_bcrs_b_w_pad, A_bcrs_col, A_bcrs_row_ptr, A_bcrs_val);
    smax->kernel("my_spmv_cuda")->register_B(A_bcrs_n_cols*A_bcrs_b_w_pad, x);
    smax->kernel("my_spmv_cuda")->register_C(A_bcrs_n_rows*A_bcrs_b_h_pad, y);
    smax->kernel("my_spmv_cuda")->set_block_column_major(block_column_major);
    smax->kernel("my_spmv_cuda")->set_bspmv_kernel_implementation(SMAX::BCRSKernelType::naive_thread_per_row);

    // Function to test
    smax->kernel("my_spmv_cuda")->run();

    compare_arrays(y, y_expected, A_bcrs_n_rows * A_bcrs_b_h_pad, "cuda_spmv_y");

    delete[] A_bcrs_col;
    delete[] A_bcrs_row_ptr;
    delete[] A_bcrs_val;
    delete[] x;
    delete[] y;
    delete[] y_expected;
    delete smax;
}