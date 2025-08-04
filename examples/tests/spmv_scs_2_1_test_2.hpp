#pragma once

//       1   2   3   4   5   6   7   8   9   10
//       _________________________________________
//  1   |11          14  15                      |
//  2   |    22                                  |
//  3   |31  32  33                              |
//  4   |            44                          |
//  5   |                55                      |
//  6   |                    66          69  610 |
//  7   |                        77              |
//  8   |                    86  87  88          |
//  9   |                                99      |
//  10  |                                    1010|
//       _________________________________________

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spmv_scs_2_1_test_2) {

    using IT = int;
    using VT = double;

    // Initialize crs operand
    CRSMatrix<IT, VT> *A_crs = new CRSMatrix<IT, VT>(10, 10, 18);
    A_crs->col = new IT[A_crs->nnz]{
        0, 3, 4, 1, 0, 1, 2, 3, 4, 5, 8, 9, 6, 5, 6, 7, 8, 9,
    };
    A_crs->row_ptr = new IT[A_crs->n_rows + 1]{
        0, 3, 4, 7, 8, 9, 12, 13, 16, 17, 18,
    };
    A_crs->val = new VT[A_crs->nnz]{
        11.0, 14.0, 15.0,  22.0, 31.0, 32.0, 33.0, 44.0, 55.0,
        66.0, 69.0, 610.0, 77.0, 86.0, 87.0, 88.0, 99.0, 1010.0,
    };

    // Declare Sell-c-sigma operand
    SCSMatrix<IT, VT> *A_scs = new SCSMatrix<IT, VT>(2, 1);

    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_scs<IT, VT, ULL>(
        A_crs->n_rows, A_crs->n_cols, A_crs->nnz, A_crs->col, A_crs->row_ptr,
        A_crs->val, A_scs->C, A_scs->sigma, A_scs->n_rows, A_scs->n_rows_padded,
        A_scs->n_cols, A_scs->n_chunks, A_scs->n_elements, A_scs->nnz,
        A_scs->chunk_ptr, A_scs->chunk_lengths, A_scs->col, A_scs->val,
        A_scs->perm);

    // Declare expected output from conversion utility
    SCSMatrix<IT, VT> *A_scs_exp = new SCSMatrix<IT, VT>(2, 1);
    A_scs_exp->n_rows = 10;
    A_scs_exp->n_rows_padded = 10;
    A_scs_exp->n_cols = 10;
    A_scs_exp->n_chunks = 5;
    A_scs_exp->n_elements = 26;
    A_scs_exp->nnz = 18;
    A_scs_exp->chunk_ptr = new IT[A_scs_exp->n_chunks + 1]{
        0, 6, 12, 18, 24, 26,
    };
    A_scs_exp->chunk_lengths = new IT[A_scs_exp->n_chunks]{3, 3, 3, 3, 1};
    A_scs_exp->col = new IT[A_scs_exp->n_elements]{
        0, 1, 3, 0, 4, 0, 0, 3, 1, 0, 2, 0, 4,
        5, 0, 8, 0, 9, 6, 5, 0, 6, 0, 7, 8, 9,
    };
    A_scs_exp->val = new VT[A_scs_exp->n_elements]{
        11.0, 22.0, 14.0, 0.0,  15.0, 0.0,  31.0, 44.0,   32.0,
        0.0,  33.0, 0.0,  55.0, 66.0, 0.0,  69.0, 0.0,    610.0,
        77.0, 86.0, 0.0,  87.0, 0.0,  88.0, 99.0, 1010.0,
    };
    A_scs_exp->perm = new IT[A_scs_exp->n_rows]{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    };

    // clang-format off
    compare_values<ULL>(A_scs->n_rows, A_scs_exp->n_rows, "n_rows");
    compare_values<ULL>(A_scs->n_rows_padded, A_scs_exp->n_rows_padded, "n_rows_padded");
    compare_values<ULL>(A_scs->n_cols, A_scs_exp->n_cols, "n_cols");
    compare_values<ULL>(A_scs->n_chunks, A_scs_exp->n_chunks, "n_chunks");
    compare_values<ULL>(A_scs->n_elements, A_scs_exp->n_elements, "n_elements");
    compare_values<ULL>(A_scs->nnz, A_scs_exp->nnz, "nnz");
    compare_arrays<IT>(A_scs->chunk_ptr, A_scs_exp->chunk_ptr, A_scs->n_chunks + 1, "chunk_ptr");
    compare_arrays<IT>(A_scs->chunk_lengths, A_scs_exp->chunk_lengths, A_scs->n_chunks, "chunk_lengths");
    compare_arrays<IT>(A_scs->col, A_scs_exp->col, A_scs->n_elements, "col");
    compare_arrays<VT>(A_scs->val, A_scs_exp->val, A_scs->n_elements, "val");
    compare_arrays<IT>(A_scs->perm, A_scs_exp->perm, A_scs->n_rows, "perm");
    // clang-format on

    delete A_crs;
    delete A_scs;
    delete A_scs_exp;
    delete smax;
}