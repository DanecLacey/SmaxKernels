#ifndef SMAX_PERMUTATION_HPP
#define SMAX_PERMUTATION_HPP

#include "../utils.hpp"
#include <queue>

namespace SMAX {

template <typename IT>
void build_symmetric_csr(IT *A_row_ptr, IT *A_col, int A_n_rows,
                         IT *&A_sym_row_ptr, IT *&A_sym_col) {

    A_sym_row_ptr = new IT[A_n_rows + 1];
    A_sym_row_ptr[0] = 0;
    IT *nnz_per_row = new IT[A_n_rows];
    bool *col_visited = new bool[A_n_rows]; // used to prevent duplicates
    bool *row_visited = new bool[A_n_rows]; // used to prevent duplicates

#pragma omp parallel for
    for (int i = 0; i < A_n_rows; ++i) {
        A_sym_row_ptr[i + 1] = (IT)0;
        nnz_per_row[i] = (IT)0;
        col_visited[i] = false;
        row_visited[i] = false;
    }

    // DL 06.05.25 NOTE: We could parallelize if performance is a problem
    // === First Pass: Count entries per row for A + A^T ===
    for (int i = 0; i < A_n_rows; ++i) {
        for (int jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            int j = A_col[jj];

            // Insert nz
            if (!col_visited[j]) {
                col_visited[j] = true;
                ++nnz_per_row[i];
            }

            // Insert symmetric nz if not on diagonal
            if (i != j && !row_visited[i]) {
                row_visited[i] = true;
                ++nnz_per_row[j];
            }
        }

        // Clear visited arrays
        for (int jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            col_visited[A_col[jj]] = false;
        }
        row_visited[i] = false;
    }

    // === Build row_ptr ===
    A_sym_row_ptr[0] = 0;
    for (int i = 0; i < A_n_rows; ++i) {
        A_sym_row_ptr[i + 1] = A_sym_row_ptr[i] + nnz_per_row[i];
    }
    A_sym_col = new IT[A_sym_row_ptr[A_n_rows]];

    // DL 06.05.25 NOTE: We could parallelize if performance is a problem
    // === Second Pass: Fill col indices ===
    for (int i = 0; i < A_n_rows; ++i) {
        int offset = A_sym_row_ptr[i];
        for (int jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            int j = A_col[jj];

            // Insert nz
            if (!col_visited[j]) {
                col_visited[j] = true;
                A_sym_col[offset++] = j;
            }

            // Insert symmetric nz if not on diagonal
            if (i != j && !row_visited[i]) {
                row_visited[i] = true;
                A_sym_col[offset++] = i;
            }
        }

        // Clear visiteds
        for (int jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            col_visited[A_col[jj]] = false;
        }
        row_visited[i] = false;
    }

    delete[] nnz_per_row;
    delete[] col_visited;
    delete[] row_visited;
}

template <typename IT>
void Utils::generate_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                          int *inv_perm) {

    // Step 1: Build symmetric structure of A + A^T
    IT *A_sym_row_ptr, *A_sym_col;
    build_symmetric_csr(A_row_ptr, A_col, A_n_rows, A_sym_row_ptr, A_sym_col);

    // Step 2: Simulate a level-order traversal (BFS)
    std::vector<bool> visited(A_n_rows, false);
    int perm_index = 0;

    // TODO: island?
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

#pragma omp parallel for
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
        for (int jj = A_row_ptr[i]; jj < A_row_ptr[i + 1]; ++jj) {
            IT j = A_col[jj];
            VT value = A_val[jj]; // Original value at (i, j)

            // Put the value in the new position
            int permuted_index =
                A_perm_row_ptr[perm_row]++; // Get the index for the
                                            // permuted row

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