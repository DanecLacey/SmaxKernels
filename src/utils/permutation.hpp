#ifndef SMAX_PERMUTATION_HPP
#define SMAX_PERMUTATION_HPP

#include "helpers.hpp"
#include <algorithm>
#include <cstdlib>
#include <queue>

namespace SMAX {

template <typename IT>
void build_symmetric_csr(IT *A_row_ptr, IT *A_col, int A_n_rows,
                         IT *&A_sym_row_ptr, IT *&A_sym_col, int &A_sym_nnz) {

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
                // insert i into row j → use A_sym_col[A_sym_row_ptr[j] + …]
                A_sym_col[A_sym_row_ptr[j] + (--nnz_per_row[j])] = i;
                // –––or alternatively build a per‐row cursor array just like we
                // did for col_perm–––
            }
        }
    }

    delete[] nnz_per_row;
    delete[] col_visited;
    delete[] sym_visited;
}


template <typename IT>
void Utils::generate_perm_jh(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm, int *inv_perm){

    int *levels = (int *) malloc(sizeof(int) * A_n_rows);
    if (levels == nullptr){
        fprintf(stderr, "Malloc not succesfull in generate_perm.\n");
        exit(EXIT_FAILURE);
    }
        
    // Step 1: Build symmetric structure of A + A^T
    IT *A_sym_row_ptr, *A_sym_col;
    int A_sym_nnz = 0;
    build_symmetric_csr(A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col,
                        A_sym_nnz);
    print_matrix<IT>(A_n_rows, A_n_rows, A_sym_nnz, A_sym_col, A_sym_row_ptr);

    int max_level = 0;

    // Step 2: Compute levels for each row in A + A^T
    for (int row_idx = 0; row_idx < A_n_rows; row_idx++){
        levels[row_idx] = 0;
        for (int nz = A_sym_row_ptr[row_idx]; nz < A_sym_row_ptr[row_idx+1]; nz++){
            if (A_sym_col[nz] < row_idx){
                levels[row_idx] = std::max(levels[row_idx], levels[A_sym_col[nz]] + 1);
                max_level = max_level < levels[row_idx]?levels[row_idx]:max_level;
            }
        }
    }

    // Step 3: Create range vector and use sorting function to permute into final permutation
    for (int i = 0; i < A_n_rows; i++){
        perm[i] = i;
    }
    std::stable_sort(perm, perm + A_n_rows, [&](const int& a, const int& b){return (levels[a] < levels[b]); });

    // Step 4: Compute inverse perm
    for (int i = 0; i < A_n_rows; ++i) {
        inv_perm[perm[i]] = i;
    }

    free(levels);

}

template <typename IT>
void Utils::generate_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                          int *inv_perm) {

    // Step 1: Build symmetric structure of A + A^T
    IT *A_sym_row_ptr, *A_sym_col;
    int A_sym_nnz = 0;
    build_symmetric_csr(A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col,
                        A_sym_nnz);
    print_matrix<IT>(A_n_rows, A_n_rows, A_sym_nnz, A_sym_col, A_sym_row_ptr);

    // Step 2: Simulate a level-order traversal (BFS)
    std::vector<bool> visited(A_n_rows, false);
    int perm_index = 0;

    // DL 07.05.25 TODO: How to account for islands?
    for (int start = 0; start < A_n_rows; ++start) {
        if (visited[start])
            continue; // Skip already visited nodes

        std::queue<int> q; // Queue for BFS
        q.push(start);
        visited[start] = true;

        // Perform BFS
        while (!q.empty()) {
            int u = q.front(); // Get the front node in the queue
            q.pop();           // Remove the front node from the queue

            perm[perm_index++] = u;

            // For each non-zero element (neighbor) v in row u
            for (int jj = A_sym_row_ptr[u]; jj < A_sym_row_ptr[u + 1]; ++jj) {
                int v = A_sym_col[jj]; // Column index represents a neighbor

                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v); // Add the neighbor to the queue
                }
            }
        }
    }

    // Step 3: Compute inverse permutation
    for (int i = 0; i < A_n_rows; ++i) {
        inv_perm[perm[i]] = i;
    }

    delete[] A_sym_row_ptr;
    delete[] A_sym_col;
};

template <typename IT, typename VT>
void Utils::apply_mat_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, VT *A_val,
                           IT *A_perm_row_ptr, IT *A_perm_col, VT *A_perm_val,
                           int *perm) {

    // Construct row_ptr for permuted mat
    A_perm_row_ptr[0] = (IT)0;

#pragma omp parallel for
    for (int i = 0; i < A_n_rows; ++i) {
        A_perm_row_ptr[i + 1] = (IT)0;
    }

    for (int i = 0; i < A_n_rows; ++i) {
        int perm_row = perm[i];

        // Count the non-zero elements in the permuted row
        for (IT jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            IT j = A_col[jj];
            ++A_perm_row_ptr[perm_row + 1];
        }
    }

    for (int i = 0; i < A_n_rows; ++i) {
        A_perm_row_ptr[i + 1] += A_perm_row_ptr[i];
    }

    for (int i = 0; i < A_n_rows; ++i) {
        int perm_row = perm[i];
        int offset = A_perm_row_ptr[perm_row];
        for (int jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            IT j = A_col[jj];
            VT value = A_val[jj]; // Original value at (i, j)

            // Put the value in the new position
            int permuted_index = offset++;

            // Store the column index and value in the permuted matrix
            A_perm_col[permuted_index] = perm[j];
            A_perm_val[permuted_index] = value;
        }
    }

    // DL 06.05.2020 TODO: Obvious candidate for loop fusion
    // #pragma omp parallel
    //     {
    // #pragma omp for
    //         for (int i = 0; i < A_n_rows; ++i) {
    //             A_perm_row_ptr[i + 1] = (IT)0;

    //             int perm_row = perm[i];

    //             // Count the non-zero elements in the permuted row
    //             for (IT jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
    //                 IT j = A_col[jj];
    //                 ++A_perm_row_ptr[perm_row + 1];
    //             }
    //         }
    // #pragma omp barrier
    // #pragma omp for
    //         for (int i = 0; i < A_n_rows; ++i) {
    //             A_perm_row_ptr[i + 1] += A_perm_row_ptr[i];
    //         }
    // #pragma omp barrier
    // #pragma omp for
    //         for (int i = 0; i < A_n_rows; ++i) {
    //             int perm_row = perm[i];
    //             for (int jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
    //                 IT j = A_col[jj];
    //                 VT value = A_val[jj]; // Original value at (i, j)

    //                 // Put the value in the new position
    //                 int permuted_index =
    //                     A_perm_row_ptr[perm_row]++; // Get the index for
    // the
    //                                                 // permuted row

    //                 // Store the column index and value in the permuted
    //                 matrix A_perm_col[permuted_index] = inv_perm[j];
    //                 A_perm_val[permuted_index] = value;
    //             }
    //         }
    //     }
};

template <typename VT>
void Utils::apply_vec_perm(int n_rows, VT *vec, VT *vec_perm, int *perm) {

#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        vec[perm[i]] = vec_perm[i];
    }
};

} // namespace SMAX

#endif // SMAX_PERMUTATION_HPP
