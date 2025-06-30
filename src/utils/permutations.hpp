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

template <typename IT>
int Utils::build_symmetric_csr(IT *A_row_ptr, IT *A_col, int A_n_rows,
                               IT *&A_sym_row_ptr, IT *&A_sym_col,
                               int &A_sym_nnz) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering build_symmetric_csr"));

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

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting build_symmetric_csr"));

    return 0;
};

template <typename IT>
int Utils::generate_perm_row_sweep(int A_n_rows, IT *A_sym_row_ptr,
                                   IT *A_sym_col, int *perm, int *inv_perm,
                                   int *lvl) {

    // suppress compiler warnings
    (void)perm;
    (void)inv_perm;

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
                             int *perm, int *inv_perm, int *lvl) {

    // suppress compiler warnings
    (void)inv_perm;

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
    if (type == "RS") {
        n_levels = Utils::generate_perm_row_sweep(
            A_n_rows, A_sym_row_ptr, A_sym_col, perm, inv_perm, lvl);
    } else if (type == "BFS") {
        n_levels = Utils::generate_perm_BFS(A_n_rows, A_sym_row_ptr, A_sym_col,
                                            perm, inv_perm, lvl);
    } else if (type == "SC") {
        n_levels = Utils::generate_color_perm(A_n_rows, A_sym_row_ptr,
                                              A_sym_col, perm, inv_perm, lvl);
    } else if (type == "PC") {
        n_levels = Utils::generate_color_perm_par(
            A_n_rows, A_sym_row_ptr, A_sym_col, perm, inv_perm, lvl);
    } else if (type == "PC_BAL") {
        n_levels = Utils::generate_color_perm_par(
            A_n_rows, A_sym_row_ptr, A_sym_col, perm, inv_perm, lvl);
        n_levels = Utils::generate_color_perm_bal(
            A_n_rows, A_sym_row_ptr, A_sym_col, perm, inv_perm, lvl, n_levels);
    } else {
        // Throwing errors in lib is not nice.
        // TODO: think of a way to tell user that the wrong type is used.
        UtilsErrorHandler::perm_type_dne(type, "BFS, RS, SC, PC, PC_BAL");
    }

    // Step 3: Compute permuation - if necessary
    if (type != "BFS") {
        for (int i = 0; i < A_n_rows; i++) {
            perm[i] = i;
        }
        std::stable_sort(
            perm, perm + A_n_rows,
            [&](const int &a, const int &b) { return (lvl[a] < lvl[b]); });
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

// Testing Coloring options
// Based on Saad Alg 3.6
template <typename IT>
int Utils::generate_color_perm(int A_n_rows, IT *A_sym_row_ptr, IT *A_sym_col,
                               int *perm, int *inv_perm, int *lvl) {

    // suppress compiler warnings
    (void)perm;
    (void)inv_perm;

    int *color = lvl;
    int *usable_colors = new int[A_n_rows];
    for (int i = 0; i < A_n_rows; i++) {
        color[i] = -2;
        usable_colors[i] = -1;
    }

    int max_color = 0;

    // Use BFS-like traversal to assign colors
    //
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
            color[start] = -1;
        }
    }

    // Perform BFS
    while (!q.empty()) {
        int u = q.front();
        q.pop();

        // Enqueue all unvisited neighbors
        for (int jj = A_sym_row_ptr[u]; jj < A_sym_row_ptr[u + 1]; ++jj) {
            int v = A_sym_col[jj];
            if (color[v] == -2) {
                color[v] = -1;
                q.push(v);
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

// Based on ``High performance and balanced parallel graph coloring on multicore
// platforms'' C. Giannoula, A. Peppas, G. Goumas, N. Koziris
template <typename IT>
int Utils::generate_color_perm_par(int A_n_rows, IT *A_sym_row_ptr,
                                   IT *A_sym_col, int *perm, int *inv_perm,
                                   int *lvl) {

    // suppress compiler warnings
    (void)perm;
    (void)inv_perm;

    int max_color = 0;
    int *colors = lvl;
    for (int i = 0; i < A_n_rows; i++) {
        colors[i] = -1;
    }

#pragma omp parallel for schedule(static, 10)
    for (int row = 0; row < A_n_rows; row++) {
        bool repeat = true;
        // Repeat until conflicts resolved
        int rep = 0;
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

// Based on ``High performance and balanced parallel graph coloring on multicore
// platforms'' C. Giannoula, A. Peppas, G. Goumas, N. Koziris
template <typename IT>
int Utils::generate_color_perm_bal(int A_n_rows, IT *A_sym_row_ptr,
                                   IT *A_sym_col, int *perm, int *inv_perm,
                                   int *lvl, int num_colors) {

    int *colors = lvl;
    double b = A_n_rows / num_colors;
    int *color_size = new int[num_colors];
    for (int c = 0; c < num_colors; c++) {
        color_size[c] = 0;
    }
    // Count size of each lvl
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
bool Utils::sanity_check_perm(const int A_n_rows, const IT *A_row_ptr, const IT *A_col, const int *colors){
    
    bool valid = true;

#pragma omp parallel for reduction(&:valid)
    for (int row = 0; row < A_n_rows; row++) {
        for (int idx = A_row_ptr[row]; idx < A_row_ptr[row + 1]; idx++) {
            if (A_col[idx] >= row)
                continue;
            if (colors[row] == colors[A_col[idx]]) {
                std::cout << "Found problem in color " << colors[row] << " with nodes " << row << " and " << A_col[idx] << std::endl;
                valid &= false;
            }
        }
    }
    
    return valid;

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
