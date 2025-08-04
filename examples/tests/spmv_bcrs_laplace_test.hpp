
#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

void test_blocked_conversion(CRSMatrix<int, double> *A_crs, bool column_major,
                             bool print_dense_matrix = false) {
    const bool use_blocked_column_major = column_major;
    const int target_b_height = 4;
    const int target_b_width = 8;
    const int target_height_pad = 4;
    const int target_width_pad = 8;

    // Declare bcrs operand
    BCRSMatrix<int, double> *A_bcrs = new BCRSMatrix<int, double>;

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_bcrs<int, double, ULL>(
        A_crs->n_rows, A_crs->n_cols, A_crs->nnz, A_crs->col, A_crs->row_ptr,
        A_crs->val, A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
        A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad, A_bcrs->b_w_pad,
        A_bcrs->col, A_bcrs->row_ptr, A_bcrs->val, target_b_height,
        target_b_width, target_height_pad, target_width_pad,
        use_blocked_column_major);

    if (print_dense_matrix) {
        std::cout << "\n--------------------------------------------\n";
        std::vector<std::string> strings(target_b_height);
        for (ULL mrow = 0; mrow < A_bcrs->n_rows; ++mrow) {
            std::for_each(strings.begin(), strings.end(),
                          [](auto &str) { str.clear(); });
            int idx = A_bcrs->row_ptr[mrow];
            for (ULL mcol = 0; mcol < A_bcrs->n_cols; ++mcol) {
                while (mcol > (ULL)A_bcrs->col[idx] &&
                       idx < A_bcrs->row_ptr[mrow + 1]) {
                    ++idx;
                }
                bool entry = (ULL)A_bcrs->col[idx] == mcol;
                for (ULL k = 0; k < A_bcrs->b_height; ++k) {
                    for (ULL l = 0; l < A_bcrs->b_width; ++l) {
                        if (entry) {
                            if (use_blocked_column_major)
                                strings.at(k) +=
                                    std::string(" ") +
                                    std::to_string(
                                        A_bcrs->val[idx * A_bcrs->b_h_pad *
                                                        A_bcrs->b_w_pad +
                                                    l * A_bcrs->b_height + k]);
                            else
                                strings.at(k) +=
                                    std::string(" ") +
                                    std::to_string(
                                        A_bcrs->val[idx * A_bcrs->b_h_pad *
                                                        A_bcrs->b_w_pad +
                                                    k * A_bcrs->b_width + l]);
                        } else
                            strings.at(k) +=
                                std::string(" ") + std::to_string(double(0));
                    }
                }
            }
            for (ULL i = 0; i < A_bcrs->b_height; ++i) {
                std::cout << strings.at(i) << "\n";
            }
        }
        std::cout << "\n--------------------------------------------\n";
    }

    double *x = new double[A_bcrs->n_cols * A_bcrs->b_w_pad];
    for (ULL i = 0; i < A_bcrs->n_cols * A_bcrs->b_width; ++i) {
        x[i] = 1.0 / double((i + 3) % 23 + 1);
    }

    // Initialize result
    double *y = new double[A_bcrs->n_rows * A_bcrs->b_h_pad];

    // Initialize expected result
    double *y_expected = new double[A_bcrs->n_rows * A_bcrs->b_h_pad];

    smax->register_kernel("my_spmv", SMAX::KernelType::SPMV);
    smax->kernel("my_spmv")->register_A(A_crs->n_rows, A_crs->n_cols,
                                        A_crs->nnz, A_crs->col, A_crs->row_ptr,
                                        A_crs->val);
    smax->kernel("my_spmv")->register_B(A_crs->n_cols, x);
    smax->kernel("my_spmv")->register_C(A_crs->n_rows, y_expected);

    // Function to test
    smax->kernel("my_spmv")->run();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("my_bspmv", SMAX::KernelType::SPMV);
    smax->kernel("my_bspmv")->set_mat_bcrs(true);
    smax->kernel("my_bspmv")
        ->register_A(A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
                     A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
                     A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr,
                     A_bcrs->val);
    smax->kernel("my_bspmv")->register_B(A_bcrs->n_cols * A_bcrs->b_w_pad, x);
    smax->kernel("my_bspmv")->register_C(A_bcrs->n_rows * A_bcrs->b_h_pad, y);
    smax->kernel("my_bspmv")->set_block_column_major(use_blocked_column_major);

    // Function to test
    smax->kernel("my_bspmv")->run();

    compare_arrays(y, y_expected, A_bcrs->n_rows * A_bcrs->b_h_pad, "spmv_y");

    // register cuda kernel and compare, if available
#if SMAX_CUDA_MODE
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
    smax->kernel("my_spmv_cuda")
        ->set_block_column_major(use_blocked_column_major);
    if (column_major) {
        smax->kernel("my_spmv_cuda")
            ->set_kernel_implementation(SMAX::SpMVType::naive_warp_shuffle);
    } else {
        smax->kernel("my_spmv_cuda")
            ->set_kernel_implementation(SMAX::SpMVType::naive_warp_group);
    }

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

REGISTER_TEST(bspmv_laplace_test) {

    // constexpr int mat_size = 1608;
    constexpr int mat_size = 16;
    // prints full matrix, only sensible for very small matrix_sizes
    bool print_dense_matrix = false;
    CRSMatrix<int, double> *A_crs = new CRSMatrix<int, double>;

    A_crs->n_rows = mat_size;
    A_crs->n_cols = mat_size;
    A_crs->nnz = 3 * mat_size - 2;
    A_crs->col = new int[A_crs->nnz];
    A_crs->row_ptr = new int[A_crs->n_rows + 1];
    A_crs->val = new double[A_crs->nnz];

    A_crs->col[0] = int(0);
    A_crs->col[1] = int(1);
    A_crs->row_ptr[0] = int(0);
    A_crs->row_ptr[1] = int(2);
    A_crs->val[0] = double(2);
    A_crs->val[1] = double(-1);
    for (ULL i = 1; i < A_crs->n_rows - 1; ++i) {
        A_crs->col[2 + (i - 1) * 3] = int(i - 1);
        A_crs->col[2 + (i - 1) * 3 + 1] = int(i);
        A_crs->col[2 + (i - 1) * 3 + 2] = int(i + 1);
        A_crs->row_ptr[i + 1] = A_crs->row_ptr[i] + int(3);
        A_crs->val[2 + (i - 1) * 3] = double(-1);
        A_crs->val[2 + (i - 1) * 3 + 1] = double(2);
        A_crs->val[2 + (i - 1) * 3 + 2] = double(-1);
    }
    A_crs->col[A_crs->nnz - 2] = int(A_crs->n_rows - 2);
    A_crs->col[A_crs->nnz - 1] = int(A_crs->n_rows - 1);
    A_crs->row_ptr[A_crs->n_rows] = int(A_crs->nnz);
    A_crs->val[A_crs->nnz - 2] = double(-1);
    A_crs->val[A_crs->nnz - 1] = double(2);

    // test_blocked_conversion(A_crs, true, print_dense_matrix);
    test_blocked_conversion(A_crs, false, print_dense_matrix);

    delete A_crs;
}
