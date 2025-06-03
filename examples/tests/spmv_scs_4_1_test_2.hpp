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

REGISTER_TEST(spmv_scs_4_1_test_2) {

    using IT = int;
    using VT = double;

    // Initialize crs operand
    int A_crs_n_rows = 10;
    int A_crs_n_cols = 10;
    int A_crs_nnz = 18;
    IT *A_crs_col = new IT[A_crs_nnz]{
        0, 3, 4, 1, 0, 1, 2, 3, 4, 5, 8, 9, 6, 5, 6, 7, 8, 9,
    };
    IT *A_crs_row_ptr = new IT[A_crs_n_rows + 1]{
        0, 3, 4, 7, 8, 9, 12, 13, 16, 17, 18,
    };
    VT *A_crs_val = new VT[A_crs_nnz]{
        11.0, 14.0, 15.0,  22.0, 31.0, 32.0, 33.0, 44.0, 55.0,
        66.0, 69.0, 610.0, 77.0, 86.0, 87.0, 88.0, 99.0, 1010.0,
    };

    // Declare Sell-c-sigma operand
    int A_scs_C = 4;     // Defined by user
    int A_scs_sigma = 1; // Defined by user
    int A_scs_n_rows = 0;
    int A_scs_n_rows_padded = 0;
    int A_scs_n_cols = 0;
    int A_scs_n_chunks = 0;
    int A_scs_n_elements = 0;
    int A_scs_nnz = 0;
    IT *A_scs_chunk_ptr = nullptr;
    IT *A_scs_chunk_lengths = nullptr;
    IT *A_scs_col = nullptr;
    VT *A_scs_val = nullptr;
    IT *A_scs_perm = nullptr;

    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_scs<IT, VT>(
        A_crs_n_rows, A_crs_n_cols, A_crs_nnz, A_crs_col, A_crs_row_ptr,
        A_crs_val, A_scs_C, A_scs_sigma, A_scs_n_rows, A_scs_n_rows_padded,
        A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements, A_scs_nnz,
        A_scs_chunk_ptr, A_scs_chunk_lengths, A_scs_col, A_scs_val, A_scs_perm);

    // Declare expected output from conversion utility
    int A_scs_n_rows_expected = 10;
    int A_scs_n_rows_padded_expected = 12;
    int A_scs_n_cols_expected = 10;
    int A_scs_n_chunks_expected = 3;
    int A_scs_n_elements_expected = 28;
    int A_scs_nnz_expected = 18;
    IT *A_scs_chunk_ptr_expected =
        new IT[A_scs_n_chunks_expected + 1]{0, 12, 24, 28};
    IT *A_scs_chunk_lengths_expected = new IT[A_scs_n_chunks_expected]{3, 3, 1};
    IT *A_scs_col_expected = new IT[A_scs_n_elements_expected]{
        0, 1, 0, 3, 3, 0, 1, 0, 4, 0, 2, 0, 4, 5,
        6, 5, 0, 8, 0, 6, 0, 9, 0, 7, 8, 9, 0, 0,
    };
    VT *A_scs_val_expected = new VT[A_scs_n_elements_expected]{
        11.0, 22.0,  31.0, 44.0, 14.0, 0.0,    32.0, 0.0,  15.0, 0.0,
        33.0, 0.0,   55.0, 66.0, 77.0, 86.0,   0.0,  69.0, 0.0,  87.0,
        0.0,  610.0, 0.0,  88.0, 99.0, 1010.0, 0.0,  0.0,
    };
    IT *A_scs_perm_expected =
        new IT[A_scs_n_rows_expected]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    compare_values<int>(A_scs_n_rows, A_scs_n_rows_expected, "n_rows");
    compare_values<int>(A_scs_n_rows_padded, A_scs_n_rows_padded_expected,
                        "n_rows_padded");
    compare_values<int>(A_scs_n_cols, A_scs_n_cols_expected, "n_cols");
    compare_values<int>(A_scs_n_chunks, A_scs_n_chunks_expected, "n_chunks");
    compare_values<int>(A_scs_n_elements, A_scs_n_elements_expected,
                        "n_elements");
    compare_values<int>(A_scs_nnz, A_scs_nnz_expected, "nnz");

    compare_arrays<IT>(A_scs_chunk_ptr, A_scs_chunk_ptr_expected,
                       A_scs_n_chunks + 1, "chunk_ptr");
    compare_arrays<IT>(A_scs_chunk_lengths, A_scs_chunk_lengths_expected,
                       A_scs_n_chunks, "chunk_lengths");
    compare_arrays<IT>(A_scs_col, A_scs_col_expected, A_scs_n_elements, "col");
    compare_arrays<VT>(A_scs_val, A_scs_val_expected, A_scs_n_elements, "val");
    compare_arrays<IT>(A_scs_perm, A_scs_perm_expected, A_scs_n_rows, "perm");

    delete[] A_crs_col;
    delete[] A_crs_row_ptr;
    delete[] A_crs_val;
    delete[] A_scs_chunk_ptr_expected;
    delete[] A_scs_chunk_lengths_expected;
    delete[] A_scs_col_expected;
    delete[] A_scs_val_expected;
    delete[] A_scs_perm_expected;
    delete smax;
}