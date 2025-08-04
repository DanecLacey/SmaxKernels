#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(bspmv_test) {

    // Initialize operands
    // Declare expected output from conversion utility
    BCRSMatrix<int, double> *A_bcrs =
        new BCRSMatrix<int, double>(2, 2, 3, 2, 2, 2, 2);
    const bool block_column_major = true;
    A_bcrs->col = new int[A_bcrs->n_blocks]{0, 1, 1};
    A_bcrs->row_ptr = new int[A_bcrs->n_rows + 1]{0, 2, 3};
    A_bcrs->val =
        new double[A_bcrs->n_blocks * A_bcrs->b_w_pad * A_bcrs->b_h_pad]{
            1.1, 0.0, 0.0, 2.2, 1.3, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 4.4};

    double *x = new double[A_bcrs->n_cols * A_bcrs->b_w_pad];
    for (ULL i = 0; i < A_bcrs->n_cols * A_bcrs->b_width; ++i) {
        x[i] = 1.0;
    }

    // Initialize result
    double *y = new double[A_bcrs->n_rows * A_bcrs->b_h_pad];

    // Initialize expected result
    double *y_expected =
        new double[A_bcrs->n_rows * A_bcrs->b_h_pad]{3.8, 2.2, 0., 4.4};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("my_spmv", SMAX::KernelType::SPMV);
    smax->kernel("my_spmv")->set_mat_bcrs(true);
    smax->kernel("my_spmv")->register_A(
        A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks, A_bcrs->b_height,
        A_bcrs->b_width, A_bcrs->b_h_pad, A_bcrs->b_w_pad, A_bcrs->col,
        A_bcrs->row_ptr, A_bcrs->val);
    smax->kernel("my_spmv")->register_B(A_bcrs->n_cols * A_bcrs->b_w_pad, x);
    smax->kernel("my_spmv")->register_C(A_bcrs->n_rows * A_bcrs->b_h_pad, y);
    smax->kernel("my_spmv")->set_block_column_major(block_column_major);

    // Function to test
    smax->kernel("my_spmv")->run();

    compare_arrays(y, y_expected, A_bcrs->n_rows * A_bcrs->b_h_pad, "spmv_y");

#if SMAX_CUDA_MODE
    // register cuda kernel and compare, if available
    smax->register_kernel("my_spmv_cuda", SMAX::KernelType::SPMV,
                          SMAX::PlatformType::CUDA);
    smax->kernel("my_spmv_cuda")->set_mat_bcrs(true);
    smax->kernel("my_spmv_cuda")
        ->register_A(A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
                     A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
                     A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr,
                     A_bcrs->val);
    smax->kernel("my_spmv_cuda")
        ->register_B(A_bcrs->n_cols * A_bcrs->b_w_pad, x);
    smax->kernel("my_spmv_cuda")
        ->register_C(A_bcrs->n_rows * A_bcrs->b_h_pad, y);

    // Give special kernel configuration
    smax->kernel("my_spmv_cuda")->set_block_column_major(block_column_major);
    smax->kernel("my_spmv_cuda")
        ->set_kernel_implementation(SMAX::SpMVType::naive_thread_per_row);

    // Function to test
    smax->kernel("my_spmv_cuda")->run();

    compare_arrays(y, y_expected, A_bcrs->n_rows * A_bcrs->b_h_pad,
                   "cuda_spmv_y");
#endif

    delete A_bcrs;
    delete[] x;
    delete[] y;
    delete[] y_expected;
    delete smax;
}