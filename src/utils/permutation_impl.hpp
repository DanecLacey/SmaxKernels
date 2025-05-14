#pragma once

#include "../common.hpp"
#include "utils_common.hpp"
#include <algorithm>
#include <cstdlib>
#include <queue>

namespace SMAX {

template <typename IT>
void build_symmetric_csr(IT *A_row_ptr, IT *A_col, int A_n_rows,
                         IT *&A_sym_row_ptr, IT *&A_sym_col, int &A_sym_nnz) {

    IF_DEBUG(ErrorHandler::log("Entering build_symmetric_csr"));

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

    IF_DEBUG(ErrorHandler::log("Exiting build_symmetric_csr"));
};

template <typename IT>
void Utils::generate_perm_jh(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                             int *inv_perm) {

    int *levels = (int *)malloc(sizeof(int) * A_n_rows);
    if (levels == nullptr) {
        fprintf(stderr, "Malloc not succesfull in generate_perm.\n");
        exit(EXIT_FAILURE);
    }

    // Step 1: Build symmetric structure of A + A^T
    IT *A_sym_row_ptr, *A_sym_col;
    int A_sym_nnz = 0;
    build_symmetric_csr(A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col,
                        A_sym_nnz);

    int max_level = 0;

    // Step 2: Compute levels for each row in A + A^T
    for (int row_idx = 0; row_idx < A_n_rows; row_idx++) {
        levels[row_idx] = 0;
        for (int nz = A_sym_row_ptr[row_idx]; nz < A_sym_row_ptr[row_idx + 1];
             nz++) {
            if (A_sym_col[nz] < row_idx) {
                levels[row_idx] =
                    std::max(levels[row_idx], levels[A_sym_col[nz]] + 1);
                max_level =
                    max_level < levels[row_idx] ? levels[row_idx] : max_level;
            }
        }
    }

    // Step 2.5: Submit level information to uc
    IF_DEBUG(
        ErrorHandler::log("%d levels detected in generate_perm", max_level));
    uc->lvl_ptr = new int[max_level + 1];

    // Count nodes per level
    int *count = new int[max_level];
    for (int i = 0; i < max_level; ++i) {
        count[i] = 0;
    }
    for (int i = 0; i < A_n_rows; ++i) {
        ++count[levels[i]];
    }

    // Build the prefix‐sum pointer array (size = max_level+1)
    uc->lvl_ptr[0] = 0;
    for (int L = 0; L < max_level; ++L) {
        uc->lvl_ptr[L + 1] = uc->lvl_ptr[L] + count[L];
    }

    uc->n_levels= max_level;

    // Step 3: Create range vector and use sorting function to permute into
    // final permutation
    for (int i = 0; i < A_n_rows; i++) {
        perm[i] = i;
    }
    std::stable_sort(perm, perm + A_n_rows, [&](const int &a, const int &b) {
        return (levels[a] < levels[b]);
    });

    // Step 4: Compute inverse perm
    for (int i = 0; i < A_n_rows; ++i) {
        inv_perm[perm[i]] = i;
    }

    free(levels);
}

template <typename IT>
void Utils::generate_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                          int *inv_perm) {

    IF_DEBUG(ErrorHandler::log("Entering generate_perm"));

    int *lvl = new int[A_n_rows];

    // Step 1: Build symmetric structure of A + A^T
    IT *A_sym_row_ptr, *A_sym_col;
    int A_sym_nnz = 0;
    build_symmetric_csr(A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col,
                        A_sym_nnz);

    // Step 2: Simulate a level-order traversal (BFS) and collect levels
    std::vector<bool> visited(A_n_rows, false);
    int perm_index = 0;
    int global_max_level = 0;

    // DL 07.05.25 TODO: How to account for islands?
    for (int start = 0; start < A_n_rows; ++start) {
        if (visited[start])
            continue; // Skip already visited nodes

        std::queue<int> q; // Queue for BFS
        q.push(start);
        visited[start] = true;

        int current_level = 0;

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
                for (int jj = A_sym_row_ptr[u]; jj < A_sym_row_ptr[u + 1];
                     ++jj) {
                    int v = A_sym_col[jj];
                    if (!visited[v]) {
                        visited[v] = true;
                        q.push(v);
                    }
                }
            }

            // Done one whole level
            ++current_level;
        }
    }

    // Compute inverse permutation
    for (int i = 0; i < A_n_rows; ++i) {
        inv_perm[perm[i]] = i;
    }

    int n_levels = global_max_level + 1;
    IF_DEBUG(
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

    IF_DEBUG(ErrorHandler::log("Exiting generate_perm"));
};

template <typename IT, typename VT>
void Utils::apply_mat_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, VT *A_val,
                           IT *A_perm_row_ptr, IT *A_perm_col, VT *A_perm_val,
                           int *perm, int *inv_perm) {

    IF_DEBUG(ErrorHandler::log("Entering apply_mat_perm"));

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

            IF_DEBUG(
                if (A_perm_col[perm_idx] >= A_n_rows) {
                    UtilsErrorHandler::col_ob(A_perm_col[perm_idx], perm_idx,
                                              A_n_rows, "apply_mat_perm");
                } if (A_perm_col[perm_idx] < 0) {
                    UtilsErrorHandler::col_ub(A_perm_col[perm_idx], perm_idx, 0,
                                              "apply_mat_perm");
                });
        }
    }

    IF_DEBUG(ErrorHandler::log("Exiting apply_mat_perm"));
};

template <typename VT>
void Utils::apply_vec_perm(int n_rows, VT *vec, VT *vec_perm, int *perm) {

    IF_DEBUG(ErrorHandler::log("Entering apply_vec_perm"));

#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        vec_perm[i] = vec[perm[i]];
    }

    IF_DEBUG(ErrorHandler::log("Exiting apply_vec_perm"));
};

} // namespace SMAX
