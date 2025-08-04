#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(bspmv_bcsr_1_1_test_2) {

    using IT = int;
    using VT = double;

    // Initialize crs operand
    CRSMatrix<IT, VT> *A_crs = new CRSMatrix<IT, VT>;
    A_crs->n_rows = 4;
    A_crs->n_cols = 4;
    A_crs->nnz = 5;
    A_crs->col = new IT[A_crs->nnz]{0, 2, 3, 1, 3};
    A_crs->row_ptr = new IT[A_crs->n_rows + 1]{0, 3, 4, 4, 5};
    A_crs->val = new VT[A_crs->nnz]{1.1, 1.3, 1.4, 2.2, 4.4};

    const bool use_blocked_column_major = false;
    const int target_b_height = 2;
    const int target_b_width = 2;
    const int target_height_pad = 2;
    const int target_width_pad = 2;

    // Declare bcrs operand
    BCRSMatrix<IT, VT> *A_bcrs = new BCRSMatrix<IT, VT>;

    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_bcrs<IT, VT, ULL>(
        A_crs->n_rows, A_crs->n_cols, A_crs->nnz, A_crs->col, A_crs->row_ptr,
        A_crs->val, A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
        A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad, A_bcrs->b_w_pad,
        A_bcrs->col, A_bcrs->row_ptr, A_bcrs->val, target_b_height,
        target_b_width, target_height_pad, target_width_pad,
        use_blocked_column_major);

    // Declare expected output from conversion utility
    BCRSMatrix<IT, VT> *A_bcrs_exp =
        new BCRSMatrix<IT, VT>(2, 2, 3, 2, 2, 2, 2);
    A_bcrs_exp->col = new IT[A_bcrs_exp->n_blocks]{0, 1, 1};
    A_bcrs_exp->row_ptr = new IT[A_bcrs_exp->n_rows + 1]{0, 2, 3};
    A_bcrs_exp->val = new VT[A_bcrs_exp->n_blocks * A_bcrs_exp->b_w_pad *
                             A_bcrs_exp->b_h_pad]{1.1, 0.0, 0.0, 2.2, 1.3, 1.4,
                                                  0.0, 0.0, 0.0, 0.0, 0.0, 4.4};

    // clang-format off
    compare_values<ULL>(A_bcrs->n_rows, A_bcrs_exp->n_rows, "n_rows");
    compare_values<ULL>(A_bcrs->n_cols, A_bcrs_exp->n_cols, "n_cols");
    compare_values<ULL>(A_bcrs->n_blocks, A_bcrs_exp->n_blocks, "n_blocks");
    compare_arrays<IT>(A_bcrs->col, A_bcrs_exp->col, A_bcrs_exp->n_blocks, "col");
    compare_arrays<IT>(A_bcrs->row_ptr, A_bcrs_exp->row_ptr, A_bcrs_exp->n_rows + 1, "row");
    compare_arrays<VT>(A_bcrs->val, A_bcrs_exp->val, A_bcrs_exp->n_blocks * A_bcrs_exp->b_h_pad * A_bcrs_exp->b_w_pad, "val");
    // clang-format on

    delete A_crs;
    delete A_bcrs;
    delete A_bcrs_exp;
    delete smax;
}