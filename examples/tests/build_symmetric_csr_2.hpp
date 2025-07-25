#pragma once

//                  Before
//        0   1   2   3   4   5   6   7
//       _______________________________
//  0   |11          14  15            |
//  1   |    22                        |
//  2   |31  32  33                    |
//  3   |            44                |
//  4   |    52          55      57    |
//  5   |61                  66        |
//  6   |                75      77    |
//  7   |                            88|
//       _______________________________

//                  After
//        0   1   2   3   4   5   6   7
//       _______________________________
//  0   |11      **  14  15  **        |
//  1   |    22  **      **            |
//  2   |31  32  33                    |
//  3   |**          44                |
//  4   |**  52          55      57    |
//  5   |61                  66        |
//  6   |                75      77    |
//  7   |                            88|
//       _______________________________

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(build_symmetric_csr_2) {

    using IT = int;

    // Initialize operands
    IT A_n_rows = 8;
    IT A_nnz = 16;
    IT *A_col = new IT[A_nnz]{0, 3, 4, 1, 0, 1, 2, 3, 1, 4, 6, 0, 5, 4, 6, 7};
    IT *A_row_ptr = new IT[A_n_rows + 1]{0, 3, 4, 7, 8, 11, 13, 15, 16};

    // Declare permuted data
    IT A_sym_n_rows = 8;
    IT A_sym_nnz = 0;
    IT *A_sym_col = nullptr;
    IT *A_sym_row_ptr = nullptr;

    // Initialize expected result
    IT expected_A_sym_nnz = 22;
    IT *expected_A_sym_col = new IT[expected_A_sym_nnz]{
        0, 2, 3, 4, 5, 1, 2, 4, 0, 1, 2, 0, 3, 0, 1, 4, 6, 0, 5, 4, 6, 7};
    IT *expected_A_sym_row_ptr =
        new IT[A_sym_n_rows + 1]{0, 5, 8, 11, 13, 17, 19, 21, 22};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Function to test
    smax->utils->build_symmetric_csr<IT, IT>(
        A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col, A_sym_nnz);

    sort_csr_rows_by_col<IT, IT>(A_sym_row_ptr, A_sym_col, A_n_rows, A_sym_nnz);

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
