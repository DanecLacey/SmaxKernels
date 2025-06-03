#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spmv_scs_2_1_test_1) {

    using IT = int;
    using VT = double;

    // Initialize crs operand
    int A_crs_n_rows = 3;
    int A_crs_n_cols = 3;
    int A_crs_nnz = 5;
    IT *A_crs_col = new IT[A_crs_nnz]{0, 1, 1, 0, 2};
    IT *A_crs_row_ptr = new IT[A_crs_n_rows + 1]{0, 2, 3, 5};
    VT *A_crs_val = new VT[A_crs_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    // Declare Sell-c-sigma operand
    int A_scs_C = 2;     // Defined by user
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
    int A_scs_n_rows_expected = 3;
    int A_scs_n_rows_padded_expected = 4;
    int A_scs_n_cols_expected = 3;
    int A_scs_n_chunks_expected = 2;
    int A_scs_n_elements_expected = 8;
    int A_scs_nnz_expected = 5;
    IT *A_scs_chunk_ptr_expected = new IT[A_scs_n_chunks_expected + 1]{0, 4, 8};
    IT *A_scs_chunk_lengths_expected = new IT[A_scs_n_chunks_expected]{2, 2};
    IT *A_scs_col_expected = new IT[A_scs_n_elements_expected]{
        0, 1, 1, 0, 0, 0, 2, 0,
    };
    VT *A_scs_val_expected = new VT[A_scs_n_elements_expected]{
        1.1, 2.2, 1.2, 0.0, 3.1, 0.0, 3.3, 0.0,
    };
    IT *A_scs_perm_expected = new IT[A_scs_n_rows_expected]{0, 1, 2};

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