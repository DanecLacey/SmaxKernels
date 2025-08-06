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

REGISTER_TEST(build_symmetric_crs_2) {

    using IT = int;
    using VT = double;

    // Initialize operands
    CRSMatrix<IT, VT> *A = new CRSMatrix<IT, VT>(8, 8, 16);
    A->col = new IT[A->nnz]{0, 3, 4, 1, 0, 1, 2, 3, 1, 4, 6, 0, 5, 4, 6, 7};
    A->row_ptr = new IT[A->n_rows + 1]{0, 3, 4, 7, 8, 11, 13, 15, 16};

    // Declare permuted data
    CRSMatrix<IT, VT> *A_sym = new CRSMatrix<IT, VT>;
    A_sym->n_rows = 8;

    // Initialize expected result
    CRSMatrix<IT, VT> *A_sym_exp =
        new CRSMatrix<IT, VT>(A_sym->n_rows, A_sym->n_rows, 22);
    A_sym_exp->col = new IT[A_sym_exp->nnz]{0, 2, 3, 4, 5, 1, 2, 4, 0, 1, 2,
                                            0, 3, 0, 1, 4, 6, 0, 5, 4, 6, 7};
    A_sym_exp->row_ptr =
        new IT[A_sym->n_rows + 1]{0, 5, 8, 11, 13, 17, 19, 21, 22};

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Function to test
    smax->utils->build_symmetric_crs<IT, ULL>(
        A->row_ptr, A->col, A->n_rows, A_sym->row_ptr, A_sym->col, A_sym->nnz);

    sort_crs_rows_by_col<IT, IT>(A_sym->row_ptr, A_sym->col, A->n_rows,
                                 A_sym->nnz);

    // Compare results
    compare_values<IT>(A_sym_exp->nnz, A_sym->nnz, std::string("nnz"));
    compare_arrays<IT>(A_sym_exp->col, A_sym->col, A_sym->nnz,
                       std::string("col"));
    compare_arrays<IT>(A_sym_exp->row_ptr, A_sym->row_ptr, A_sym->n_rows + 1,
                       std::string("row_ptr"));

    delete A;
    delete A_sym;
    delete A_sym_exp;
    delete smax;
}
