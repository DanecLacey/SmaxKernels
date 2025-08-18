#pragma once

#include "../common.hpp"
#include "utils_common.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <queue>
#include <unordered_set>

namespace SMAX {

template <typename IT, typename ST>
int build_symmetric_crs_old(IT *A_row_ptr, IT *A_col, int A_n_rows,
                            IT *&A_sym_row_ptr, IT *&A_sym_col, ST &A_sym_nnz) {
    A_sym_row_ptr = new IT[A_n_rows + 1];
    A_sym_row_ptr[0] = (IT)0;
    IT *nnz_per_row = new IT[A_n_rows];
    bool *col_visited = new bool[A_n_rows]; // to prevent duplicates
    bool *sym_visited = new bool[A_n_rows];
    std::vector<IT> touched_cols;
    std::vector<IT> touched_sym;

#pragma omp parallel for
    for (int i = 0; i < A_n_rows; ++i) {
        A_sym_row_ptr[i + 1] = (IT)0;
        nnz_per_row[i] = (IT)0;
        col_visited[i] = false;
        sym_visited[i] = false;
    }

    // === First pass: Counting ===
    for (int i = 0; i < A_n_rows; ++i) {
        touched_cols.clear();
        touched_sym.clear();
        for (IT jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            int j = A_col[jj];

            // original entry in row i
            if (!col_visited[j]) {
                col_visited[j] = true;
                ++nnz_per_row[i];
                touched_cols.push_back(j);
            }

            // symmetric entry in row j
            if (i != j && !sym_visited[j]) {

                // binary‐search row j for i
                auto bj = A_col + A_row_ptr[j];
                auto ej = A_col + A_row_ptr[j + 1];
                if (!std::binary_search(bj, ej, i)) {
                    sym_visited[j] = true;
                    ++nnz_per_row[j];
                    touched_sym.push_back(j);
                }
            }
        }

        // Reset only touched entries
        for (int j : touched_cols) {
            col_visited[j] = false;
        }
        for (int j : touched_sym) {
            sym_visited[j] = false;
        }
    }

    A_sym_row_ptr[0] = 0;
    for (int i = 0; i < A_n_rows; ++i) {
        A_sym_row_ptr[i + 1] = A_sym_row_ptr[i] + nnz_per_row[i];
    }
    A_sym_nnz = A_sym_row_ptr[A_n_rows];
    A_sym_col = new IT[A_sym_nnz];

    // Just to be extra safe
#pragma omp parallel for
    for (int i = 0; i < A_n_rows; ++i) {
        col_visited[i] = false;
        sym_visited[i] = false;
    }

    // === Second pass: Assignment ===
    for (int i = 0; i < A_n_rows; ++i) {
        touched_cols.clear();
        touched_sym.clear();
        int offset = A_sym_row_ptr[i];
        for (IT jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            int j = A_col[jj];

            // original
            if (!col_visited[j]) {
                col_visited[j] = true;
                A_sym_col[offset++] = j;
                touched_cols.push_back(j);
            }

            // symmetric
            if (i != j && !sym_visited[j]) {
                auto bj = A_col + A_row_ptr[j];
                auto ej = A_col + A_row_ptr[j + 1];
                if (!std::binary_search(bj, ej, i)) {
                    sym_visited[j] = true;
                    A_sym_col[A_sym_row_ptr[j] + (--nnz_per_row[j])] = i;
                    touched_sym.push_back(j);
                }
            }

            // Reset only touched entries
            for (int j : touched_cols) {
                col_visited[j] = false;
            }
            for (int j : touched_sym) {
                sym_visited[j] = false;
            }
        }
    }

    // === Third pass: Sorting == NOTE: Optional
#pragma omp parallel for
    for (int i = 0; i < A_n_rows; ++i) {
        auto begin = A_sym_col + A_sym_row_ptr[i];
        auto end = A_sym_col + A_sym_row_ptr[i + 1];
        std::sort(begin, end);
    }

    delete[] nnz_per_row;
    delete[] col_visited;
    delete[] sym_visited;

    return 0;
}

template <typename IT, typename ST>
int build_symmetric_crs_parallel(IT *A_row_ptr, IT *A_col, int A_n_rows,
                                 IT *&A_sym_row_ptr, IT *&A_sym_col,
                                 ST &A_sym_nnz) {

    // Based on RACE::makeSymmetricGraph() at https://github.com/RRZE-HPC/RACE
    A_sym_nnz = A_row_ptr[A_n_rows];
    std::vector<std::vector<int>> new_col_in_row(A_n_rows);

    // Check if matrix is symmetric
    volatile bool is_symmetric = true;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < A_n_rows; ++r) {
        for (IT j = A_row_ptr[r]; j < A_row_ptr[r + 1]; ++j) {
            IT c = A_col[j];
            if (c != r && c >= 0 && c < A_n_rows) { // Assuming square matrix!
                bool has_pair = false;
                for (int k = A_row_ptr[c]; k < A_row_ptr[c + 1]; ++k) {
                    if (A_col[k] == r) {
                        has_pair = true;
                        break;
                    }
                }
                if (!has_pair) {
                    // add the missing pair
                    is_symmetric = false;
                    // do an update to the rows and col
#pragma omp critical
                    {
                        new_col_in_row[c].push_back(r);
                        ++A_sym_nnz;
                    }
                }
            }
        } // loop over col in row
    } // loop over row

    A_sym_row_ptr = new IT[A_n_rows + 1];
    A_sym_col = new IT[A_sym_nnz];

    if (is_symmetric) {
        // TODO: Just copy the pointers
        // A_sym_row_ptr = A_row_ptr;
        // A_sym_col = A_col;
        A_sym_nnz = A_row_ptr[A_n_rows];

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int i = 0; i < A_n_rows + 1; ++i) {
                A_sym_row_ptr[i] = A_row_ptr[i];
            }
#pragma omp for schedule(static)
            for (int i = 0; i < A_sym_nnz; ++i) {
                A_sym_col[i] = A_col[i];
            }
        }
    } else {
        // Create the new data structure adding the missing entries

        // update rowPtr
        A_sym_row_ptr[0] = A_row_ptr[0];

        // NUMA init
#pragma omp parallel for schedule(static)
        for (int r = 0; r < A_n_rows; ++r) {
            A_sym_row_ptr[r + 1] = (IT)0;
        }
        for (int r = 0; r < A_n_rows; ++r) {
            // new rowLen = old + extra nnz;
            int row_len =
                (A_row_ptr[r + 1] - A_row_ptr[r]) + new_col_in_row[r].size();
            A_sym_row_ptr[r + 1] = A_sym_row_ptr[r] + row_len;
        }

        if (A_sym_nnz != A_sym_row_ptr[A_n_rows]) {
            IF_SMAX_DEBUG(ErrorHandler::log("New nnz count does not match the "
                                            "last entry in A_sym_row_ptr"));
            return 1;
        }

        // update col
#pragma omp parallel for schedule(static)
        for (int r = 0; r < A_n_rows; ++r) {
            IT j, old_j;
            for (j = A_sym_row_ptr[r], old_j = A_row_ptr[r];
                 old_j < A_row_ptr[r + 1]; ++j, ++old_j) {
                A_sym_col[j] = A_col[old_j];
            }
            // add extra nnzs now
            ULL ctr = 0;
            for (; j < A_sym_row_ptr[r + 1]; ++j) {
                A_sym_col[j] = new_col_in_row[r][ctr];
                ++ctr;
            }
        }
    }

    return 0;
}

template <typename IT, typename ST>
int build_symmetric_crs_parallel_adapted(IT *A_row_ptr, IT *A_col, int A_n_rows,
                                         IT *&A_sym_row_ptr, IT *&A_sym_col,
                                         ST &A_sym_nnz) {

    // Based on RACE::makeSymmetricGraph() at https://github.com/RRZE-HPC/RACE

    A_sym_nnz = 0;
    int threads = 1;
#pragma omp parallel
    {
        threads = omp_get_num_threads();
    }

    std::vector<std::vector<int>> **thread_storage =
        new std::vector<std::vector<int>>
            *[threads]; // (std::vector<std::vector<int>> **)malloc(threads *
                        // sizeof(std::vector<std::vector<int>>*));
#pragma omp parallel
    {
        std::vector<std::vector<int>> *local_vector =
            new std::vector<std::vector<int>>(A_n_rows);
        thread_storage[omp_get_thread_num()] = local_vector;
    }

    // Check if matrix is symmetric
    bool is_symmetric = true;
#pragma omp parallel for schedule(static) reduction(& : is_symmetric)          \
    reduction(+ : A_sym_nnz)
    for (int r = 0; r < A_n_rows; ++r) {
        int t_num = omp_get_thread_num();
        for (IT j = A_row_ptr[r]; j < A_row_ptr[r + 1]; ++j) {
            IT c = A_col[j];
            if (c != r && c >= 0 && c < A_n_rows) { // Assuming square matrix!
                bool has_pair = false;
                for (int k = A_row_ptr[c]; k < A_row_ptr[c + 1]; ++k) {
                    if (A_col[k] == r) {
                        has_pair = true;
                        break;
                    }
                }
                if (!has_pair) {
                    // add the missing pair
                    is_symmetric &= false;
                    // do an update to the rows and col
                    (*thread_storage[t_num])[c].push_back(r);
                    A_sym_nnz += 1;
                }
            }
        } // loop over col in row
    } // loop over row

    A_sym_nnz += A_row_ptr[A_n_rows];

    A_sym_row_ptr = new IT[A_n_rows + 1];
    A_sym_col = new IT[A_sym_nnz];

    if (is_symmetric) {
        // TODO: Just copy the pointers
        // A_sym_row_ptr = A_row_ptr;
        // A_sym_col = A_col;
        A_sym_nnz = A_row_ptr[A_n_rows];

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int i = 0; i < A_n_rows + 1; ++i) {
                A_sym_row_ptr[i] = A_row_ptr[i];
            }
#pragma omp for schedule(static)
            for (int i = 0; i < A_sym_nnz; ++i) {
                A_sym_col[i] = A_col[i];
            }
        }
    } else {
        // Create the new data structure adding the missing entries

        // update rowPtr
        A_sym_row_ptr[0] = A_row_ptr[0];

        int *sizes = new int[A_n_rows];
#pragma omp parallel for schedule(static)
        for (int r = 0; r < A_n_rows; r++) {
            sizes[r] = 0;
        }

        // TODO: Make parallel
        for (int r = 0; r < A_n_rows; r++) {
            for (int t = 0; t < threads; t++) {
                sizes[r] += (*thread_storage[t])[r].size();
            }
        }

        // NUMA init
#pragma omp parallel for schedule(static)
        for (int r = 0; r < A_n_rows; ++r) {
            A_sym_row_ptr[r + 1] = (IT)0;
        }
        for (int r = 0; r < A_n_rows; ++r) {
            int row_len = (A_row_ptr[r + 1] - A_row_ptr[r]) + sizes[r];
            A_sym_row_ptr[r + 1] = A_sym_row_ptr[r] + row_len;
        }

        if (A_sym_nnz != A_sym_row_ptr[A_n_rows]) {
            IF_SMAX_DEBUG(ErrorHandler::log("New nnz count does not match the "
                                            "last entry in A_sym_row_ptr"));
            return 1;
        }

        // update col
#pragma omp parallel for schedule(static)
        for (int r = 0; r < A_n_rows; ++r) {
            IT j, old_j;
            for (j = A_sym_row_ptr[r], old_j = A_row_ptr[r];
                 old_j < A_row_ptr[r + 1]; ++j, ++old_j) {
                A_sym_col[j] = A_col[old_j];
            }
            // add extra nnzs now
            ULL thread_counter = 0;
            ULL thread_local_ctr = 0;
            for (; j < A_sym_row_ptr[r + 1]; ++j) {
                while (thread_local_ctr >=
                       (*thread_storage[thread_counter])[r].size()) {
                    thread_local_ctr = 0;
                    thread_counter += 1;
                }
                A_sym_col[j] =
                    (*thread_storage[thread_counter])[r][thread_local_ctr];
                thread_local_ctr += 1;
            }
        }
        delete[] sizes;
    }
    delete[] thread_storage;
    return 0;
}

template <typename IT, typename ST>
int Utils::build_symmetric_crs(IT *A_row_ptr, IT *A_col, ST A_n_rows,
                               IT *&A_sym_row_ptr, IT *&A_sym_col,
                               ST &A_sym_nnz) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering build_symmetric_crs"));
#if 0
    build_symmetric_crs_old<IT, ST>(A_row_ptr, A_col, A_n_rows, A_sym_row_ptr,
                                    A_sym_col, A_sym_nnz);
#elif 0
    build_symmetric_crs_parallel<IT, ST>(A_row_ptr, A_col, A_n_rows,
                                         A_sym_row_ptr, A_sym_col, A_sym_nnz);
#elif 1
    build_symmetric_crs_parallel_adapted<IT, ST>(
        A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col, A_sym_nnz);
#endif
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting build_symmetric_crs"));

    return 0;
};

template <typename IT>
int Utils::generate_perm_row_sweep(int A_n_rows, IT *A_sym_row_ptr,
                                   IT *A_sym_col, int *lvl) {

    int max_level = 0;

    // Compute levels for each row in A + A^T
    for (int row_idx = 0; row_idx < A_n_rows; row_idx++) {
        lvl[row_idx] = 0;
        for (int nz = A_sym_row_ptr[row_idx]; nz < A_sym_row_ptr[row_idx + 1];
             nz++) {
            if (A_sym_col[nz] < row_idx) {
                lvl[row_idx] = std::max(lvl[row_idx], lvl[A_sym_col[nz]] + 1);
                max_level = max_level < lvl[row_idx] ? lvl[row_idx] : max_level;
            }
        }
    }

    // Step 2.5: Submit level information to uc
    return ++max_level; // increase by one since levels are 0 indexed
}

template <typename IT>
int Utils::generate_perm_BFS(int A_n_rows, IT *A_sym_row_ptr, IT *A_sym_col,
                             int *perm, int *lvl) {

    std::vector<bool> visited(A_n_rows, false);

    // Start queueing the island roots
    std::queue<int> q; // Queue for BFS
#pragma omp parallel for schedule(static, 10)
    for (int start = 0; start < A_n_rows; ++start) {
        bool is_root = true;
        for (int nnz = A_sym_row_ptr[start]; nnz < A_sym_row_ptr[start + 1];
             ++nnz) {
            if (A_sym_col[nnz] < start) {
                is_root = false;
                break;
            }
        }
#pragma omp critical
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
#pragma omp parallel for
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
#pragma omp critical
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
int Utils::generate_perm_BFS_BW(int A_n_rows, IT *A_sym_row_ptr, IT *A_sym_col,
                                int *perm, int *lvl) {

    std::vector<bool> visited(A_n_rows, false);
    std::queue<int> q;

    int perm_index = 0;
    int global_max_level = -1;

    for (int start = 0; start < A_n_rows; ++start) {
        if (visited[start])
            continue;

        // start new component
        visited[start] = true;
        lvl[start] = 0;
        q.push(start);
        global_max_level = std::max(global_max_level, 0);

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            int ulevel = lvl[u];

            // record permutation (caller must ensure perm has size >= A_n_rows)
            perm[perm_index++] = u;

            // explore neighbors using the CSR arrays passed in
            for (IT jj = A_sym_row_ptr[u]; jj < A_sym_row_ptr[u + 1]; ++jj) {
                // cast/validate column index
                int v = static_cast<int>(A_sym_col[jj]);
                if (v < 0 || v >= A_n_rows)
                    continue; // guard against corrupt input

                if (!visited[v]) {
                    visited[v] = true;
                    lvl[v] = ulevel + 1;
                    global_max_level = std::max(global_max_level, lvl[v]);
                    q.push(v);
                }
            }
        }
    }

    // global_max_level == -1 only if A_n_rows==0, which we handled earlier
    return global_max_level + 1;
};

template <typename IT>
void Utils::generate_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                          int *inv_perm, std::string type) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering generate_perm"));

    // Step 0: Make array for level information
    int *lvl = new int[A_n_rows];
#pragma omp parallel for
    for (int i = 0; i < A_n_rows; ++i) {
        lvl[i] = -1;
    }

    // Step 1: Build symmetric structure of A + A^T
    IT *A_sym_row_ptr, *A_sym_col;
    int A_sym_nnz = 0;
    build_symmetric_crs(A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col,
                        A_sym_nnz);

    // Step 2: Call kernel for desired traversal method
    int n_levels = 1;
    if (type == "RS") {
        n_levels = Utils::generate_perm_row_sweep(A_n_rows, A_sym_row_ptr,
                                                  A_sym_col, lvl);
    } else if (type == "BFS") {
        n_levels = Utils::generate_perm_BFS(A_n_rows, A_sym_row_ptr, A_sym_col,
                                            perm, lvl);
    } else if (type == "BFS_BW") {
        n_levels = Utils::generate_perm_BFS_BW(A_n_rows, A_sym_row_ptr,
                                               A_sym_col, perm, lvl);
    } else if (type == "SC") {
        n_levels =
            Utils::generate_color_perm(A_n_rows, A_sym_row_ptr, A_sym_col, lvl);
    } else if (type == "PC") {
        n_levels = Utils::generate_color_perm_par(A_n_rows, A_sym_row_ptr,
                                                  A_sym_col, lvl);
    } else if (type == "PC_BAL") {
        n_levels = Utils::generate_color_perm_par(A_n_rows, A_sym_row_ptr,
                                                  A_sym_col, lvl);
        n_levels = Utils::generate_color_perm_bal(A_n_rows, A_sym_row_ptr,
                                                  A_sym_col, lvl, n_levels);
    } else if (type == "NONE") {
        // Generates dummy permutation
#pragma omp parallel for
        for (int i = 0; i < A_n_rows; ++i) {
            lvl[i] = i;
        }
        n_levels = A_n_rows;
    } else {
        // Throwing errors in lib is not nice.
        // TODO: think of a way to tell user that the wrong type is
        // used.
        UtilsErrorHandler::perm_type_dne(
            type, "BFS, BFS_BW, RS, SC, PC, PC_BAL, NONE");
    }

    if (type != "RACE") {
        // Step 3: Compute permuation - if necessary
        if (type != "BFS") {
#pragma omp parallel for schedule(static, 10)
            for (int i = 0; i < A_n_rows; i++) {
                perm[i] = i;
            }
            std::stable_sort(
                perm, perm + A_n_rows,
                [&](const int &a, const int &b) { return (lvl[a] < lvl[b]); });
        }

        // Compute inverse permutation
#pragma omp parallel for schedule(static, 10)
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

        if (type == "BFS_BW") {
            IF_SMAX_DEBUG(
                ErrorHandler::log("The generated permutation is %s",
                                  (sanity_check_perm_bw(A_n_rows, A_sym_row_ptr,
                                                        A_sym_col, perm)
                                       ? "valid"
                                       : "not valid"));

                ErrorHandler::log("The generated permutation is %s",
                                  (sanity_check_perm_bw(A_n_rows, A_sym_row_ptr,
                                                        A_sym_col, inv_perm)
                                       ? "valid"
                                       : "not valid")););
        } else {
            IF_SMAX_DEBUG(ErrorHandler::log(
                "The generated permutation is %s",
                (sanity_check_perm(A_n_rows, A_sym_row_ptr, A_sym_col, lvl)
                     ? "valid"
                     : "not valid")));
        }
    }

    uc->n_levels = n_levels;
    delete[] A_sym_row_ptr;
    delete[] A_sym_col;
    delete[] lvl;
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting generate_perm"));
}

// Testing Coloring options
// Based on Saad Alg 3.6
template <typename IT>
int Utils::generate_color_perm(int A_n_rows, IT *A_sym_row_ptr, IT *A_sym_col,
                               int *lvl) {

    int *color = lvl;
    int *usable_colors = new int[A_n_rows];

    int max_color = 0;

    // Use BFS-like traversal to assign colors
    // Start queueing the island roots
    std::queue<int> q; // Queue for BFS
#pragma omp parallel for schedule(static, 10)
    for (int start = 0; start < A_n_rows; ++start) {
        color[start] = -2;
        usable_colors[start] = -1;
        bool is_root = true;
        for (int nnz = A_sym_row_ptr[start]; nnz < A_sym_row_ptr[start + 1];
             ++nnz) {
            if (A_sym_col[nnz] < start) {
                is_root = false;
                break;
            }
        }
#pragma omp critical
        if (is_root) {
            q.push(start);
            color[start] = -1;
        }
    }

    // Perform BFS
    while (!q.empty()) {
        int u = q.front();
        q.pop();

        // Enqueue all unvisited neighbors
#pragma omp parallel for
        for (int jj = A_sym_row_ptr[u]; jj < A_sym_row_ptr[u + 1]; ++jj) {
            int v = A_sym_col[jj];
            if (color[v] == -2) {
                color[v] = -1;
#pragma omp critical
                {
                    q.push(v);
                }
            } else if (color[v] >= 0) {
                usable_colors[color[v]] = u;
            }
        }

        for (int c = 0; c < A_n_rows; c++) {
            if (usable_colors[c] != u) {
                color[u] = c;
                max_color = std::max(max_color, c);
                break;
            }
        }
    }

    return ++max_color;
}

template <typename IT>
int Utils::generate_color_RACE(int A_n_rows, IT *A_row_ptr, IT *A_col,
                               int *perm, int *inv_perm) {
#ifdef SMAX_USE_RACE
    SMAX_GET_THREAD_COUNT(int, n_threads);
    // clang-format off
    RACE::Interface* race_interface = new RACE::Interface(
        A_n_rows,
        n_threads,
        RACE::POWER,
        reinterpret_cast<int *>(A_row_ptr),
        reinterpret_cast<int *>(A_col),
        false,
        1, // threads per core
        RACE::FILL,
        NULL,
        NULL
    );
    // clang-format on

    int cache_size = 1;    // TODO
    int n_repetitions = 1; // TODO

    race_interface->RACEColor(n_repetitions, 1, cache_size * 1024 * 1024);
    race_interface->getPerm(&perm, NULL);
    race_interface->getInvPerm(&inv_perm, NULL);
#else
    printf("SMAX_USE_RACE=OFF\n");
    exit(EXIT_FAILURE);
#endif
}

// Based on ``High performance and balanced parallel graph coloring on
// multicore platforms'' C. Giannoula, A. Peppas, G. Goumas, N. Koziris
template <typename IT>
int Utils::generate_color_perm_par(int A_n_rows, IT *A_sym_row_ptr,
                                   IT *A_sym_col, int *lvl) {

    int max_color = 0;
    int *colors = lvl;
#pragma omp parallel for schedule(static, 10)
    for (int i = 0; i < A_n_rows; i++) {
        colors[i] = -1;
    }

#pragma omp parallel for schedule(static, 10)
    for (int row = 0; row < A_n_rows; row++) {
        bool repeat = true;
        // Repeat until conflicts resolved
        while (repeat) {
            std::unordered_set<int> forbidden;
            std::unordered_set<int> critical;
            for (int idx = A_sym_row_ptr[row]; idx < A_sym_row_ptr[row + 1];
                 idx++) {
                int col_idx = A_sym_col[idx];
                forbidden.insert(colors[col_idx]);
                if ((((int)(col_idx / 10) % omp_get_num_threads()) !=
                     omp_get_thread_num())) {
                    critical.insert(col_idx);
                }
            }
            int spec_color = 0;
            while (forbidden.count(spec_color)) {
                spec_color++;
            }
            if (critical.empty()) {
                colors[row] = spec_color;
                repeat = false;
#pragma omp critical
                {
                    max_color = std::max(spec_color, max_color);
                }
            } else {
                // BEGIN CRITICAL
#pragma omp critical
                {
                    bool valid = true;
                    for (const auto &crit : critical) {
                        if (colors[crit] == spec_color) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        colors[row] = spec_color;
                        max_color = std::max(spec_color, max_color);
                        repeat = false;
                    }
                    // END CRITICAL
                }
            }
        }
    }

    return ++max_color;
}

// Based on ``High performance and balanced parallel graph coloring on
// multicore platforms'' C. Giannoula, A. Peppas, G. Goumas, N. Koziris
template <typename IT>
int Utils::generate_color_perm_bal(int A_n_rows, IT *A_sym_row_ptr,
                                   IT *A_sym_col, int *lvl, int num_colors) {

    int *colors = lvl;
    double b = A_n_rows / num_colors;
    int *color_size = new int[num_colors];
    for (int c = 0; c < num_colors; c++) {
        color_size[c] = 0;
    }
    // Count size of each lvl
#pragma omp parallel for schedule(static, 10)
    for (int idx = 0; idx < A_n_rows; idx++) {
        color_size[colors[idx]]++;
    }

#pragma omp parallel for schedule(static, 10)
    for (int row = 0; row < A_n_rows; row++) {
        // Skip if color class is already balanced
        if (color_size[lvl[row]] <= b)
            continue;
        bool repeat = true;
        while (repeat) {
            std::unordered_set<int> forbidden;
            std::unordered_set<int> critical;
            for (int idx = A_sym_row_ptr[row]; idx < A_sym_row_ptr[row + 1];
                 idx++) {
                int col_idx = A_sym_col[idx];
                forbidden.insert(colors[col_idx]);
                if ((((int)(col_idx / 10) % omp_get_num_threads()) !=
                     omp_get_thread_num())) {
                    critical.insert(col_idx);
                }
            }
            int spec_color = 0;
            while (forbidden.count(spec_color) || color_size[spec_color] >= b) {
                spec_color++;
            }
            if (spec_color >= num_colors) {
                break;
            }
            if (critical.empty()) {
                // Atomic operations
#pragma omp atomic
                color_size[colors[row]]--;
                colors[row] = spec_color;
#pragma omp atomic
                color_size[colors[row]]++;
                repeat = false;
            } else {
                // BEGIN CRITICAL
#pragma omp critical
                {
                    bool valid = true;
                    for (const auto &crit : critical) {
                        if (colors[crit] == spec_color) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        // Atomic operations
                        color_size[colors[row]]--;
                        colors[row] = spec_color;
                        color_size[colors[row]]++;
                        repeat = false;
                    }
                    // END CRITICAL
                }
            }
        }
    }

    return num_colors;
}

template <typename IT>
bool Utils::sanity_check_perm(const int A_n_rows, const IT *A_row_ptr,
                              const IT *A_col, const int *colors) {

    bool valid = true;

#pragma omp parallel for reduction(& : valid)
    for (int row = 0; row < A_n_rows; row++) {
        for (int idx = A_row_ptr[row]; idx < A_row_ptr[row + 1]; idx++) {
            if (A_col[idx] >= row)
                continue;
            if (colors[row] == colors[A_col[idx]] &&
                (colors[row] != 0 &&
                 colors[A_col[idx]] != 0)) // Protect against NONE case
            {
                std::cout << "Found problem in color " << colors[row]
                          << " with nodes " << row << " and " << A_col[idx]
                          << std::endl;
                valid &= false;
            }
        }
    }

    return valid;
}

template <typename IT>
bool Utils::sanity_check_perm_bw(const int A_n_rows, const IT *A_row_ptr,
                                 const IT *A_col, const int *perm) {

    /* seen[v] == 1 means value v already appeared in perm[] */
    bool *seen = new bool[A_n_rows];
    for (ULL i = 0; i < A_n_rows; ++i) {
        seen[i] = false;
    }

    for (int i = 0; i < A_n_rows; ++i) {
        int p = perm[i];
        if (p < 0 || p >= A_n_rows) {
            std::ostringstream oss;
            oss << "perm element: '" << p << "' at index '" << i
                << "' does not exist.";
            ErrorHandler::log(oss.str());
            delete[] seen;
            return false;
        }
        if (seen[p]) {
            std::ostringstream oss;
            oss << "perm element: '" << p << "' at index '" << i
                << "' duplicated.";
            ErrorHandler::log(oss.str());
            delete[] seen;
            return false;
        }
        seen[p] = true;
    }

    /* If we reached here, every entry was in-range and seen exactly once */
    delete[] seen;

    return true;
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

#pragma omp parallel for schedule(static, 10)
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

template <typename IT, typename VT>
void Utils::level_aware_copy(IT *src_row_ptr, IT *dest_row_ptr, IT *src_col,
                             IT *dest_col, VT *src_val, VT *dest_val) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering level_aware_copy"));

    dest_row_ptr[0] = src_row_ptr[0];
    for (int lvl_idx = 0; lvl_idx < uc->n_levels; lvl_idx++) {
#pragma omp parallel for schedule(static)
        for (int row = uc->lvl_ptr[lvl_idx]; row < uc->lvl_ptr[lvl_idx + 1];
             row++) {
            dest_row_ptr[row + 1] = src_row_ptr[row + 1];
            for (int idx = src_row_ptr[row]; idx < src_row_ptr[row + 1];
                 idx++) {
                dest_val[idx] = src_val[idx];
                dest_col[idx] = src_col[idx];
            }
        }
    }

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting level_aware_copy"));
}

} // namespace SMAX
