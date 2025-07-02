#pragma once
// clang-format off
//        0   1   2   3   4   5   6   7   8   9 
//       ______________________________________
//  0   |11                                    |
//  1   |21  22                                |
//  2   |31      33                            |
//  3   |41          44                        |
//  4   |    52          55                    |
//  5   |            64      66                |
//  6   |                    76  77            |
//  7   |        83              87  88        |
//  8   |                95              99    |
//  9   |                            08  09  00|
//      | _____________________________________|

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(perm_test_2) {

    using IT = int;
    using VT = double;

    // Initialize operands
    IT A_n_rows = 10;
    IT A_n_cols = 10;
    IT A_nnz = 21;
    IT *A_col = new IT[A_nnz]{0, 0, 1, 0, 2, 0, 3, 1, 4, 3, 5, 5, 6, 2, 6, 7, 4, 8, 7, 8, 9};
    IT *A_row_ptr = new IT[A_n_rows + 1]{0, 1, 3, 5, 7, 9, 11, 13, 16, 18, 21};
    VT *A_val = new VT[A_nnz]{11, 21, 22, 31, 33, 41, 44, 52, 55, 64, 66, 
                                76, 77, 83, 87, 88, 95, 99, 8, 9, 0};

    // Declare permuted data
    IT *A_bfs_col = new IT[A_nnz];
    IT *A_bfs_row_ptr = new IT[A_n_rows + 1];
    VT *A_bfs_val = new VT[A_nnz];

    IT *A_rs_col = new IT[A_nnz];
    IT *A_rs_row_ptr = new IT[A_n_rows + 1];
    VT *A_rs_val = new VT[A_nnz];

    // Declare permutation vectors
    IT *perm_bfs = new IT[A_n_rows];
    IT *inv_perm_bfs = new IT[A_n_rows];

    IT *perm_rs = new IT[A_n_rows];
    IT *inv_perm_rs = new IT[A_n_rows];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

#ifdef _OPENMP
    int t = 0;
#pragma omp parallel
    {
#pragma omp single
        {
            t = omp_get_num_threads();
        }
    }
    omp_set_num_threads(1);
#endif
    smax->utils->generate_perm<IT>(A_n_rows, A_row_ptr, A_col, perm_bfs,
                                   inv_perm_bfs, std::string("BFS"));
#ifdef _OPENMP
    omp_set_num_threads(t);
#endif

    int expected_max_level = 6;
    int *expected_levels = new int[7]{0, 1, 4, 6, 8, 9, 10};
    compare_values<IT>(smax->get_n_levels(), expected_max_level, "BFS number of levels");
    int *bfs_levels = new int[smax->get_n_levels() + 1];
    for (int i = 0; i <= smax->get_n_levels(); i++){
        bfs_levels[i] = smax->get_level_ptr_at(i);
    }
    compare_arrays<IT>(expected_levels, bfs_levels, expected_max_level + 1, "BFS level sizes");

    smax->utils->generate_perm<IT>(A_n_rows, A_row_ptr, A_col, perm_rs,
                                   inv_perm_rs, std::string("RS"));

    compare_values<IT>(smax->get_n_levels(), expected_max_level, "RS number of levels");
    int *rs_levels = new int[smax->get_n_levels() + 1];
    for (int i = 0; i <= smax->get_n_levels(); i++){
        rs_levels[i] = smax->get_level_ptr_at(i);
    }
    compare_arrays<IT>(expected_levels, rs_levels, expected_max_level + 1, "RS level sizes");

     // printf("BFS Permutation:\n");
     // print_vector<IT>(perm_bfs, A_n_rows);

     // printf("RS Permutation:\n");
     // print_vector<IT>(perm_rs, A_n_rows);

    // Compare with expected permutation vectors
    // If comparison fails, throw std::runtime_error("description")
    int *expected_perm_bfs = new int[A_n_cols]{0, 1, 2, 3, 4, 5, 8, 6, 7, 9};
    int *expected_perm_rs = new int[A_n_cols]{0, 1, 2, 3, 4, 5, 6, 8, 7, 9};

    compare_arrays<IT>(expected_perm_bfs, perm_bfs, A_n_rows, "BFS level sets");
    compare_arrays<IT>(expected_perm_rs, perm_rs, A_n_rows, "RS level sets");

    // Apply permutations to A
    smax->utils->apply_mat_perm<IT, VT>(A_n_rows, A_row_ptr, A_col, A_val,
                                        A_bfs_row_ptr, A_bfs_col, A_bfs_val,
                                        perm_bfs, inv_perm_bfs);

    smax->utils->apply_mat_perm<IT, VT>(A_n_rows, A_row_ptr, A_col, A_val,
                                        A_rs_row_ptr, A_rs_col, A_rs_val,
                                        perm_rs, inv_perm_rs);

    VT *expected_val_bfs = new VT[A_nnz]{11, 21, 22, 31, 33, 41, 44, 52, 55, 64, 66, 95, 99, 76, 77, 83, 87, 88, 8, 9, 0};
    VT *expected_val_rs = new VT[A_nnz]{11, 21, 22, 31, 33, 41, 44, 52, 55, 64, 66, 76, 77, 95, 99, 83, 87, 88, 8, 9, 0};

    compare_arrays(expected_val_bfs, A_bfs_val, A_nnz, "BFS actual permutation");
    compare_arrays(expected_val_rs, A_rs_val, A_nnz, "RS actual permutation");

    // Compare with expected matrices

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] A_bfs_col;
    delete[] A_bfs_row_ptr;
    delete[] A_bfs_val;
    delete[] A_rs_col;
    delete[] A_rs_row_ptr;
    delete[] A_rs_val;

    delete[] perm_bfs;
    delete[] inv_perm_bfs;
    delete[] perm_rs;
    delete[] inv_perm_rs;
    delete[] expected_perm_bfs;
    delete[] expected_perm_rs;
    delete[] expected_val_bfs;
    delete[] expected_val_rs;
    delete[] bfs_levels;
    delete[] rs_levels;

    delete smax;
    // clang-format on
}
