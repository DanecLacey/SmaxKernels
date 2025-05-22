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

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

#include "test_framework.hpp"
#include <stdexcept>

REGISTER_TEST(perm_test_1) {

    using IT = int;
    using VT = double;

    // Initialize operands
    IT A_n_rows = 8;
    IT A_n_cols = 8;
    IT A_nnz = 16;
    IT *A_col = new IT[A_nnz]{0, 0, 1, 1, 2, 1, 3, 1, 3, 4, 0, 5, 6, 2, 3, 7};
    IT *A_row_ptr = new IT[A_n_rows + 1]{0, 1, 3, 5, 7, 10, 12, 13, 17};
    VT *A_val = new VT[A_nnz]{11, 21, 22, 32, 33, 42, 44, 52, 54, 55,
                              61, 66, 77, 83, 84, 88};

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

    int expected_max_level = 4;
    int *expected_levels = new int[5]{0, 2, 4, 6, 8};
    if(smax->get_uc_n_levels() != expected_max_level)
        throw std::runtime_error("Number of levels does not match in BFS.");
    for (int l = 0; l <= expected_max_level; l++){
        if (expected_levels[l] != smax->get_uc_level_ptr_at(l))
            throw std::runtime_error("Level size does not match in BFS.");
    }

    smax->utils->generate_perm<IT>(A_n_rows, A_row_ptr, A_col, perm_jh,
                                   inv_perm_jh, std::string("JH"));

    if(smax->get_uc_n_levels() != expected_max_level)
        throw std::runtime_error("Number of levels does not match in JH.");
    for (int l = 0; l <= expected_max_level; l++){
        if (expected_levels[l] != smax->get_uc_level_ptr_at(l))
            throw std::runtime_error("Level size does not match in JH.");
    }

    smax->utils->generate_perm<IT>(A_n_rows, A_row_ptr, A_col, perm_dfs,
                                   inv_perm_dfs, std::string("DFS"));

    if(smax->get_uc_n_levels() != expected_max_level)
        throw std::runtime_error("Number of levels does not match in DFS.");
    for (int l = 0; l <= expected_max_level; l++){
        if (expected_levels[l] != smax->get_uc_level_ptr_at(l))
            throw std::runtime_error("Level size does not match in DFS.");
    }

    // printf("BFS Permutation:\n");
    // print_vector<IT>(perm_bfs, A_n_rows);

    // printf("JH Permutation:\n");
    // print_vector<IT>(perm_jh, A_n_rows);

    // printf("DFS Permutation:\n");
    // print_vector<IT>(perm_dfs, A_n_rows);

    // TODO: Compare with expected permutation vectors
    // If comparison fails, throw std::runtime_error("description")
    int *expected_perm = new int[A_n_cols]{0, 6, 1, 5, 2, 3, 4, 7};

    bool check_bfs = true;
    bool check_jh = true;
    bool check_dfs = true;
    bool found_jh = false, found_bfs = false, found_dfs = false;

    for (int level = 0; level < expected_max_level; level++){
        for (int idx = expected_levels[level]; idx < expected_levels[level + 1]; idx++){
            int expected_row = expected_perm[idx];
            for (int idy = expected_levels[level]; idy < expected_levels[level + 1]; idy++){
                if (perm_bfs[idy] == expected_perm[idx]){
                    found_bfs = true;
                }
                if (perm_dfs[idy] == expected_perm[idx]){
                    found_dfs = true;
                }
                if (perm_jh[idy] == expected_perm[idx]){
                    found_jh = true;
                }
            }
            if (!found_bfs)
                check_bfs = false;
            if (!found_dfs)
                check_dfs = false;
            if (!found_jh)
                check_jh = false;
        }
    }

    if (!check_bfs){
        throw std::runtime_error("Computation of BFS permutation failed.\n");
    }
    if (!check_dfs){
        throw std::runtime_error("Computation of DFS permutation failed.\n");
    }
    if (!check_jh){
        throw std::runtime_error("Computation of JH permutation failed.\n");
    }

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
    check_bfs = true, check_dfs = true, check_jh = true;
    for (int row_idx = 0; row_idx < A_n_rows; row_idx++){
        for (int idx = A_row_ptr[row_idx]; idx < A_row_ptr[row_idx + 1]; idx++){
            if (A_col[idx] != perm_bfs[A_bfs_col[A_bfs_row_ptr[inv_perm_bfs[row_idx]] + idx - A_row_ptr[row_idx]]] ||
                    A_val[idx] != A_bfs_val[A_bfs_row_ptr[inv_perm_bfs[row_idx]] + idx - A_row_ptr[row_idx]] ){
                check_bfs = false;
            }
            if (A_col[idx] != perm_dfs[A_dfs_col[A_dfs_row_ptr[inv_perm_dfs[row_idx]] + idx - A_row_ptr[row_idx]]] ||
                    A_val[idx] != A_dfs_val[A_dfs_row_ptr[inv_perm_dfs[row_idx]] + idx - A_row_ptr[row_idx]] ){
                check_dfs = false;
            }
            if (A_col[idx] != perm_jh[A_jh_col[A_jh_row_ptr[inv_perm_jh[row_idx]] + idx - A_row_ptr[row_idx]]] ||
                    A_val[idx] != A_jh_val[A_jh_row_ptr[inv_perm_jh[row_idx]] + idx - A_row_ptr[row_idx]] ){
                check_jh = false;
            }
        }
    }

    if (!check_bfs){
        throw std::runtime_error("Applying BFS permutation failed.\n");
    }
    if (!check_dfs){
        throw std::runtime_error("Applying DFS permutation failed.\n");
    }
    if (!check_jh){
        throw std::runtime_error("Applying JH permutation failed.\n");
    }


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
