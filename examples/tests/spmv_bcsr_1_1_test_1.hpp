#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(bspmv_bcsr_1_1_test_1) {

    using IT = int;
    using VT = double;

    // Initialize crs operand
    int A_crs_n_rows = 4;
    int A_crs_n_cols = 4;
    int A_crs_nnz = 5;
    IT *A_crs_col = new IT[A_crs_nnz]{0, 2, 3, 1, 3};
    IT *A_crs_row_ptr = new IT[A_crs_n_rows + 1]{0, 3, 4, 4, 5};
    VT *A_crs_val = new VT[A_crs_nnz]{1.1, 1.3, 1.4, 2.2, 4.4};


    // todo: test with 1d laplace matrix
    // int A_crs_n_rows = 1602;
    // int A_crs_n_cols = 1602;
    // int A_crs_nnz = 3*1602 -2;
    // IT *A_crs_col = new IT[A_crs_nnz];
    // IT *A_crs_row_ptr = new IT[A_crs_n_rows + 1];
    // VT *A_crs_val = new VT[A_crs_nnz];

    // A_crs_col[0] = IT(0);
    // A_crs_col[1] = IT(1);
    // A_crs_row_ptr[0] = IT(0);
    // A_crs_row_ptr[1] = IT(2);
    // A_crs_val[0] = VT(2);
    // A_crs_val[1] = VT(-1);
    // for(int i = 1; i < A_crs_n_rows-1; ++i)
    // {
    //   A_crs_col[2+(i-1)*3] = IT(i-1);
    //   A_crs_col[2+(i-1)*3+1] = IT(i);
    //   A_crs_col[2+(i-1)*3+2] = IT(i+1);
    //   A_crs_row_ptr[i+1] = A_crs_row_ptr[i] + IT(3);
    //   A_crs_val[2+(i-1)*3] = VT(-1);
    //   A_crs_val[2+(i-1)*3+1] = VT(2);
    //   A_crs_val[2+(i-1)*3+2] = VT(-1);
    // }
    // A_crs_col[A_crs_nnz-2] = IT(A_crs_n_rows-2);
    // A_crs_col[A_crs_nnz-1] = IT(A_crs_n_rows-1);
    // A_crs_row_ptr[A_crs_n_rows] = IT(A_crs_nnz);
    // A_crs_val[A_crs_nnz-2] = VT(-1);
    // A_crs_val[A_crs_nnz-1] = VT(2);

    const bool use_blocked_column_major = true;
    const int target_b_height = 2;
    const int target_b_width = 2;
    const int target_height_pad = 2;
    const int target_width_pad = 2;

    // Declare bcrs operand
    int A_bcrs_n_rows = 0;
    int A_bcrs_n_cols = 0;
    int A_bcrs_b_width = 0;
    int A_bcrs_b_height = 0;
    int A_bcrs_b_w_pad = 0;
    int A_bcrs_b_h_pad = 0;
    int A_bcrs_nnz = 0;
    IT *A_bcrs_col = nullptr;
    IT *A_bcrs_row_ptr = nullptr;
    VT *A_bcrs_val = nullptr;

    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_bcrs<IT, VT, int>(
        A_crs_n_rows, A_crs_n_cols, A_crs_nnz, A_crs_col, A_crs_row_ptr,
        A_crs_val, A_bcrs_n_rows, A_bcrs_n_cols, A_bcrs_nnz, A_bcrs_b_height, A_bcrs_b_width, A_bcrs_b_h_pad, A_bcrs_b_w_pad,
        A_bcrs_col, A_bcrs_row_ptr, A_bcrs_val, target_b_height, target_b_width, target_height_pad, target_width_pad, use_blocked_column_major);

    // Declare expected output from conversion utility
    int A_bcrs_n_rows_expected = 2;
    int A_bcrs_n_cols_expected = 2;
    int A_bcrs_b_width_expected = 2;
    int A_bcrs_b_height_expected = 2;
    int A_bcrs_b_w_pad_expected = 2;
    int A_bcrs_b_h_pad_expected = 2;
    int A_bcrs_nnz_expected = 3;
    IT *A_bcrs_col_expected = new IT[A_bcrs_nnz_expected]{0, 1, 1};
    IT *A_bcrs_row_ptr_expected = new IT[A_bcrs_n_rows_expected + 1]{0, 2, 3};
    VT *A_bcrs_val_expected = new VT[A_bcrs_nnz_expected*A_bcrs_b_w_pad_expected*A_bcrs_b_h_pad_expected]{1.1, 0.0, 0.0, 2.2, 1.3, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 4.4};

    compare_values<int>(A_bcrs_n_rows, A_bcrs_n_rows_expected, "n_rows");
    compare_values<int>(A_bcrs_n_cols, A_bcrs_n_cols_expected, "n_cols");
    compare_values<int>(A_bcrs_nnz, A_bcrs_nnz_expected, "nnz");

    compare_arrays<IT>(A_bcrs_col, A_bcrs_col_expected, A_bcrs_nnz_expected, "col");
    compare_arrays<IT>(A_bcrs_row_ptr, A_bcrs_row_ptr_expected, A_bcrs_n_rows_expected+1, "row");
    compare_arrays<VT>(A_bcrs_val, A_bcrs_val_expected, A_bcrs_nnz_expected * A_bcrs_b_h_pad_expected * A_bcrs_b_w_pad_expected, "val");

    delete[] A_crs_col;
    delete[] A_crs_row_ptr;
    delete[] A_crs_val;
    delete[] A_bcrs_col;
    delete[] A_bcrs_row_ptr;
    delete[] A_bcrs_val;
    delete[] A_bcrs_col_expected;
    delete[] A_bcrs_row_ptr_expected;
    delete[] A_bcrs_val_expected;
    delete smax;
}