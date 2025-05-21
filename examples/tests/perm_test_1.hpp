//        0   1   2   3   4   5   6   7
//       _______________________________
//  0   |11                            |
//  1   |    22                        |
//  2   |31      33                    |
//  3   |    42  43  44                |
//  4   |        53      55            |
//  5   |            64      66        |
//  6   |                75      77    |
//  7   |                    86      88|
//       _______________________________

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

#include "test_framework.hpp"

REGISTER_TEST(perm_test_1) {

    using IT = int;
    using VT = double;

    // Initialize operands
    IT A_n_rows = 8;
    IT A_n_cols = 8;
    IT A_nnz = 15;
    IT *A_col = new IT[A_nnz]{0, 1, 0, 2, 1, 2, 3, 2, 4, 3, 5, 4, 6, 5, 7};
    IT *A_row_ptr = new IT[A_n_rows + 1]{0, 1, 2, 4, 7, 9, 11, 13, 15};
    VT *A_val = new VT[A_nnz]{11.0, 22.0, 31.0, 33.0, 42.0, 43.0, 44.0, 53.0,
                              55.0, 64.0, 66.0, 75.0, 77.0, 86.0, 88.0};

    // Declare permuted data
    IT *A_bfs_col = new IT[A_nnz];
    IT *A_bfs_row_ptr = new IT[A_n_rows + 1];
    VT *A_bfs_val = new VT[A_nnz];

    IT *A_jh_col = new IT[A_nnz];
    IT *A_jh_row_ptr = new IT[A_n_rows + 1];
    VT *A_jh_val = new VT[A_nnz];

    IT *A_dfs_col = new IT[A_nnz];
    IT *A_dfs_row_ptr = new IT[A_n_rows + 1];
    VT *A_dfs_val = new VT[A_nnz];

    // Declare permutation vectors
    IT *perm_bfs = new IT[A_n_rows];
    IT *inv_perm_bfs = new IT[A_n_rows];

    IT *perm_jh = new IT[A_n_rows];
    IT *inv_perm_jh = new IT[A_n_rows];

    IT *perm_dfs = new IT[A_n_rows];
    IT *inv_perm_dfs = new IT[A_n_rows];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->generate_perm<IT>(A_n_rows, A_row_ptr, A_col, perm_bfs,
                                   inv_perm_bfs, std::string("BFS"));

    smax->utils->generate_perm<IT>(A_n_rows, A_row_ptr, A_col, perm_jh,
                                   inv_perm_jh, std::string("JH"));

    smax->utils->generate_perm<IT>(A_n_rows, A_row_ptr, A_col, perm_dfs,
                                   inv_perm_dfs, std::string("DFS"));

    // printf("BFS Permutation:\n");
    // print_vector<IT>(perm_bfs, A_n_rows);

    // printf("JH Permutation:\n");
    // print_vector<IT>(perm_jh, A_n_rows);

    // printf("DFS Permutation:\n");
    // print_vector<IT>(perm_dfs, A_n_rows);

    // TODO: Compare with expected permutation vectors
    // If comparison fails, throw std::runtime_error("description")

    // Apply permutations to A
    smax->utils->apply_mat_perm<IT, VT>(A_n_rows, A_row_ptr, A_col, A_val,
                                        A_bfs_row_ptr, A_bfs_col, A_bfs_val,
                                        perm_bfs, inv_perm_bfs);

    smax->utils->apply_mat_perm<IT, VT>(A_n_rows, A_row_ptr, A_col, A_val,
                                        A_jh_row_ptr, A_jh_col, A_jh_val,
                                        perm_jh, inv_perm_jh);

    smax->utils->apply_mat_perm<IT, VT>(A_n_rows, A_row_ptr, A_col, A_val,
                                        A_dfs_row_ptr, A_dfs_col, A_dfs_val,
                                        perm_dfs, inv_perm_dfs);

    // TODO: Compare with expected matrices
    // If comparison fails, throw std::runtime_error("description")

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] A_bfs_col;
    delete[] A_bfs_row_ptr;
    delete[] A_bfs_val;
    delete[] A_jh_col;
    delete[] A_jh_row_ptr;
    delete[] A_jh_val;
    delete[] A_dfs_col;
    delete[] A_dfs_row_ptr;
    delete[] A_dfs_val;
    delete[] perm_bfs;
    delete[] inv_perm_bfs;
    delete[] perm_jh;
    delete[] inv_perm_jh;
    delete[] perm_dfs;
    delete[] inv_perm_dfs;
    delete smax;
}
