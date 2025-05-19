#pragma once

#include "../common.hpp"
#include "utils_common.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>

namespace SMAX {

template <typename IT>
void build_symmetric_csr(IT *A_row_ptr, IT *A_col, int A_n_rows,
                         IT *&A_sym_row_ptr, IT *&A_sym_col, int &A_sym_nnz) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering build_symmetric_csr"));

    A_sym_row_ptr = new IT[A_n_rows + 1];
    A_sym_row_ptr[0] = 0;
    IT *nnz_per_row = new IT[A_n_rows];
    bool *col_visited = new bool[A_n_rows]; // used to prevent duplicates
    bool *sym_visited = new bool[A_n_rows]; // used to prevent duplicates

#pragma omp parallel for
    for (int i = 0; i < A_n_rows; ++i) {
        A_sym_row_ptr[i + 1] = (IT)0;
        nnz_per_row[i] = (IT)0;
        col_visited[i] = false;
        sym_visited[i] = false;
    }

    // === First pass: counting ===
    for (int i = 0; i < A_n_rows; ++i) {
        for (IT jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            int j = A_col[jj];

            // original entry in row i
            if (!col_visited[j]) {
                col_visited[j] = true;
                ++nnz_per_row[i];
            }

            // symmetric entry in row j
            if (i != j && !sym_visited[j]) {
                sym_visited[j] = true;
                ++nnz_per_row[j];
            }
        }

        // TODO: Clearly not scalable
        for (int i = 0; i < A_n_rows; ++i) {
            col_visited[i] = false;
            sym_visited[i] = false;
        }
    }

    // === Build row_ptr ===
    A_sym_row_ptr[0] = 0;
    for (int i = 0; i < A_n_rows; ++i) {
        A_sym_row_ptr[i + 1] = A_sym_row_ptr[i] + nnz_per_row[i];
    }
    A_sym_nnz = A_sym_row_ptr[A_n_rows];
    A_sym_col = new IT[A_sym_nnz];

    // === Second pass: assignment ===
    for (int i = 0; i < A_n_rows; ++i) {
        // TODO: Clearly not scalable
        for (int i = 0; i < A_n_rows; ++i) {
            col_visited[i] = false;
            sym_visited[i] = false;
        }

        int offset = A_sym_row_ptr[i];
        for (IT jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            int j = A_col[jj];

            // original
            if (!col_visited[j]) {
                col_visited[j] = true;
                A_sym_col[offset++] = j;
            }

            // symmetric
            if (i != j && !sym_visited[j]) {
                sym_visited[j] = true;
                A_sym_col[A_sym_row_ptr[j] + (--nnz_per_row[j])] = i;
            }
        }
    }

    delete[] nnz_per_row;
    delete[] col_visited;
    delete[] sym_visited;

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting build_symmetric_csr"));
};

template <typename IT>
int Utils::generate_perm_jh(int A_n_rows, IT *A_sym_row_ptr, IT *A_sym_col, int *perm,
                             int *inv_perm, int *lvl) {

    int max_level = 0;

    // Compute levels for each row in A + A^T
    for (int row_idx = 0; row_idx < A_n_rows; row_idx++) {
        lvl[row_idx] = 0;
        for (int nz = A_sym_row_ptr[row_idx]; nz < A_sym_row_ptr[row_idx + 1];
             nz++) {
            if (A_sym_col[nz] < row_idx) {
                lvl[row_idx] =
                    std::max(lvl[row_idx], lvl[A_sym_col[nz]] + 1);
                max_level =
                    max_level < lvl[row_idx] ? lvl[row_idx] : max_level;
            }
        }
    }

    // Step 2.5: Submit level information to uc
    return ++max_level; // increase by one since levels are 0 indexed
}

template <typename IT>
int Utils::generate_perm_DFS(int A_n_rows, IT *A_sym_row_ptr, IT *A_sym_col, int *perm,
                              int *inv_perm, int *lvl) {

    printf("Using DFS to generate permutations will not work currently. There "
           "are known bugs...\n");

    // Simulate a DFS traversal and collect levels
    std::vector<bool> visited(A_n_rows, false);
    int global_max_level = 0;

    // Start queueing the island roots
    std::deque<int> q; // Queue for DFS
    for (int start = 0; start < A_n_rows; ++start) {
        bool is_root = true;
        for (int nnz = A_sym_row_ptr[start]; nnz < A_sym_row_ptr[start + 1];
             ++nnz) {
            if (A_sym_col[nnz] < start) {
                is_root = false;
                break;
            }
        }
        if (is_root) {
            q.push_back(start);
            lvl[start] = 0;
            visited[start] = true;
        }
    }

    // TODO: JH fix DFS
    while (!q.empty()) {

        int u = q.front();
        q.pop_front();

        // Enqueue all unvisited neighbors
        // for (int jj = A_sym_row_ptr[u]; jj < A_sym_row_ptr[u + 1];
        for (int jj = A_sym_row_ptr[u + 1] - 1; jj >= A_sym_row_ptr[u]; --jj) {
            int v = A_sym_col[jj];
            if (!visited[v] && (v > u)) {
                visited[v] = true;
                q.push_front(v);
                lvl[v] = lvl[u] + 1;
                global_max_level = std::max(global_max_level, lvl[v]);
            } else if (visited[v] && (v < u)) {
                lvl[v] = lvl[u] + 1;
                global_max_level = std::max(global_max_level, lvl[v]);
            }
        }
    }

    return  global_max_level + 1;
}

template <typename IT>
int Utils::generate_perm_BFS(int A_n_rows, IT *A_sym_row_ptr, IT *A_sym_col, int *perm,
                          int *inv_perm, int *lvl) {

    std::vector<bool> visited(A_n_rows, false);

    // Start queueing the island roots
    std::queue<int> q; // Queue for BFS
    for (int start = 0; start < A_n_rows; ++start) {
        bool is_root = true;
        for (int nnz = A_sym_row_ptr[start]; nnz < A_sym_row_ptr[start + 1];
             ++nnz) {
            if (A_sym_col[nnz] < start) {
                is_root = false;
                break;
            }
        }
        if (is_root) {
            q.push(start);
            visited[start] = true;
        }
    }

    int current_level = 0;
    int global_max_level = 0;
    int perm_index = 0;

    // Perform BFS
    while (!q.empty()) {
        int level_size = int(q.size());

        for (int i = 0; i < level_size; ++i) {
            int u = q.front();
            q.pop();

            // Record in perm & level
            perm[perm_index++] = u;
            lvl[u] = current_level;
            global_max_level = std::max(global_max_level, current_level);

            // Enqueue all unvisited neighbors
            for (int jj = A_sym_row_ptr[u]; jj < A_sym_row_ptr[u + 1]; ++jj) {
                int v = A_sym_col[jj];
                bool has_dependecy = false;
                for (int jjj = A_sym_row_ptr[v]; jjj < A_sym_row_ptr[v + 1];
                     ++jjj) {
                    if ((A_sym_col[jjj] < v) && (lvl[A_sym_col[jjj]] == -1)) {
                        has_dependecy = true;
                        break;
                    }
                }
                if (!visited[v] && !has_dependecy) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }

        // Done one whole level
        ++current_level;
    }

    return global_max_level + 1;
};

template <typename IT>
void Utils::generate_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                          int *inv_perm, std::string type) {


    IF_SMAX_DEBUG(ErrorHandler::log("Entering generate_perm"));

    // Step 0: Make array for level information
    int *lvl = new int[A_n_rows];
    for (int i = 0; i < A_n_rows; ++i) {
        lvl[i] = -1;
    }

    // Step 1: Build symmetric structure of A + A^T
    IT *A_sym_row_ptr, *A_sym_col;
    int A_sym_nnz = 0;
    build_symmetric_csr(A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col,
                        A_sym_nnz);



    // Step 2: Call kernel for desired traversal method
    int n_levels = 0;
    if ( type == "JH" ){
        n_levels = Utils::generate_perm_jh(A_n_rows, A_sym_row_ptr, A_sym_col, perm, inv_perm, lvl);
    }
    else if ( type == "BFS" ){
        n_levels = Utils::generate_perm_BFS(A_n_rows, A_sym_row_ptr, A_sym_col, perm, inv_perm, lvl);
    }
    else if ( type == "DFS" ){
        n_levels = Utils::generate_perm_DFS(A_n_rows, A_sym_row_ptr, A_sym_col, perm, inv_perm, lvl);
    }
    else{
        // Throwing errors in lib is not nice.
        // TODO: think of a way to tell user that the wrong type is used.
    }
    
    // Step 3: Compute permuation - if necessary
    if ( type != "BFS" ){
        for (int i = 0; i < A_n_rows; i++) {
            perm[i] = i;
        }
        std::stable_sort(perm, perm + A_n_rows, [&](const int &a, const int &b) {
            return (lvl[a] < lvl[b]);
        });
    }

    // Compute inverse permutation
    for (int i = 0; i < A_n_rows; ++i) {
        inv_perm[perm[i]] = i;
    }

    IF_SMAX_DEBUG(
        ErrorHandler::log("%d levels detected in generate_perm", n_levels));
    uc->lvl_ptr = new int[n_levels + 1];

    // Count nodes per level
    int *count = new int[n_levels];
    for (int i = 0; i < n_levels; ++i) {
        count[i] = 0;
    }
    for (int i = 0; i < A_n_rows; ++i) {
        ++count[lvl[i]];
    }

    // Build the prefix‐sum pointer array (size = n_levels+1)
    uc->lvl_ptr[0] = 0;
    for (int L = 0; L < n_levels; ++L) {
        uc->lvl_ptr[L + 1] = uc->lvl_ptr[L] + count[L];
    }

    uc->n_levels = n_levels;
    delete[] A_sym_row_ptr;
    delete[] A_sym_col;
    delete[] lvl;
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting generate_perm"));

}

template <typename IT, typename VT>
void Utils::apply_mat_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, VT *A_val,
                           IT *A_perm_row_ptr, IT *A_perm_col, VT *A_perm_val,
                           int *perm, int *inv_perm) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering apply_mat_perm"));

    A_perm_row_ptr[0] = (IT)0;
    int perm_idx = 0;

    for (int row = 0; row < A_n_rows; ++row) {
        IT perm_row = perm[row];
        for (int idx = A_row_ptr[perm_row]; idx < A_row_ptr[perm_row + 1];
             ++idx) {
            ++perm_idx;
        }
        A_perm_row_ptr[row + 1] = perm_idx;
    }

#pragma omp parallel for schedule(static)
    for (int row = 0; row < A_n_rows; ++row) {
        IT perm_row = perm[row];

        for (int perm_idx = A_perm_row_ptr[row], idx = A_row_ptr[perm_row];
             perm_idx < A_perm_row_ptr[row + 1]; ++idx, ++perm_idx) {

            A_perm_col[perm_idx] = inv_perm[A_col[idx]];
            A_perm_val[perm_idx] = A_val[idx];

            IF_SMAX_DEBUG(
                if (A_perm_col[perm_idx] >= A_n_rows) {
                    UtilsErrorHandler::col_ob(A_perm_col[perm_idx], perm_idx,
                                              A_n_rows, "apply_mat_perm");
                } if (A_perm_col[perm_idx] < 0) {
                    UtilsErrorHandler::col_ub(A_perm_col[perm_idx], perm_idx, 0,
                                              "apply_mat_perm");
                });
        }
    }

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting apply_mat_perm"));
};

template <typename VT>
void Utils::apply_vec_perm(int n_rows, VT *vec, VT *vec_perm, int *perm) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering apply_vec_perm"));

#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        vec_perm[i] = vec[perm[i]];
    }

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting apply_vec_perm"));
};

} // namespace SMAX
