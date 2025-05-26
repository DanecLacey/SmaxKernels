#pragma once

//                  Before
//        0   1   2   3   4   5   6   7
//       _______________________________
//  0   |11                            |
//  1   |21  22                        |
//  2   |    32  33                    |
//  3   |    42      44                |
//  4   |    52      54  55            |
//  5   |61                  66        |
//  6   |                        77    |
//  7   |        83  84              88|
//       _______________________________

//                  After
//        0   1   2   3   4   5   6   7
//       _______________________________
//  0   |11  **              **        |
//  1   |21  22  **  **  **            |
//  2   |    32  33                  **|
//  3   |    42      44  **          **|
//  4   |    52      54  55            |
//  5   |61                  66        |
//  6   |                        77    |
//  7   |        83  84              88|
//       _______________________________

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(build_symmetric_csr_1) {

    using IT = int;
    using VT = double;

    // Initialize operands
    IT A_n_rows = 8;
    IT A_n_cols = 8;
    IT A_nnz = 16;
    IT *A_col = new IT[A_nnz]{0, 0, 1, 1, 2, 1, 3, 1, 3, 4, 0, 5, 6, 2, 3, 7};
    IT *A_row_ptr = new IT[A_n_rows + 1]{0, 1, 3, 5, 7, 10, 12, 13, 16};

    // Declare permuted data
    IT A_sym_n_rows = 8;
    IT A_sym_n_cols = 8;
    IT A_sym_nnz = 0;
    IT *A_sym_col = nullptr;
    IT *A_sym_row_ptr = nullptr;

    // Initialize expected result
    IT expected_A_sym_nnz = 24;
    IT *expected_A_sym_col = new IT[expected_A_sym_nnz]{
        0, 1, 5, 0, 1, 2, 3, 4, 1, 2, 7, 1, 3, 4, 7, 1, 3, 4, 0, 5, 6, 2, 3, 7};
    IT *expected_A_sym_row_ptr =
        new IT[A_sym_n_rows + 1]{0, 3, 8, 11, 15, 18, 20, 21, 24};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Function to test
    smax->utils->build_symmetric_csr<IT>(A_row_ptr, A_col, A_n_rows,
                                         A_sym_row_ptr, A_sym_col, A_sym_nnz);

    // print_array<IT>(A_sym_col, A_sym_nnz, std::string("sym_col"));
    // print_array<IT>(A_sym_row_ptr, A_sym_n_rows + 1,
    //                 std::string("sym_row_ptr"));

    // Compare results
    compare_values<IT>(expected_A_sym_nnz, A_sym_nnz, std::string("nnz"));
    compare_arrays<IT>(expected_A_sym_col, A_sym_col, A_sym_nnz,
                       std::string("col"));
    compare_arrays<IT>(expected_A_sym_row_ptr, A_sym_row_ptr, A_sym_n_rows + 1,
                       std::string("row_ptr"));

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_sym_col;
    delete[] A_sym_row_ptr;
    delete[] expected_A_sym_col;
    delete[] expected_A_sym_row_ptr;
    delete smax;
}
