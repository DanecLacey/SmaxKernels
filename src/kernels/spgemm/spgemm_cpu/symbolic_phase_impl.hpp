#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spgemm_common.hpp"

namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU {

template <typename IT, typename VT>
inline void
padded_symbolic_phase(int A_n_rows, IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                      int B_n_rows, int B_n_cols, IT *RESTRICT B_col,
                      IT *RESTRICT B_row_ptr, int &C_n_rows, int &C_n_cols,
                      int &C_nnz, IT *&C_col, IT *&C_row_ptr, VT *&C_val) {

    GET_THREAD_COUNT(int, n_threads);
    if (n_threads > 1) {
        SpGEMMErrorHandler::multithreaded_issue();
    }

    // Enforce dimensions of C
    C_n_rows = A_n_rows;
    C_n_cols = B_n_cols;
    C_nnz = 0;

    // Count nnz in each row of B
    int *B_nnz_per_row = new int[B_n_rows];
#pragma omp parallel for schedule(static)
    for (int i = 0; i < B_n_rows; ++i) {
        B_nnz_per_row[i] = B_row_ptr[i + 1] - B_row_ptr[i];
    }

    // Collect upper-bound offsets for each thread
    int *tl_ub = new int[n_threads];
    int *tl_offsets = new int[n_threads + 1];

#pragma omp parallel
    {
        GET_THREAD_ID(int, tid)
        tl_ub[tid] = 0;

#pragma omp for schedule(static)
        for (int i = 0; i < A_n_rows; ++i) {
            int local_ub = 0;
            for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
                IT col = A_col[j];
                local_ub += B_nnz_per_row[col];
            }
            tl_ub[tid] += local_ub;
        }
    }

    tl_offsets[0] = 0;
    for (int tid = 1; tid < n_threads + 1; ++tid) {
        tl_offsets[tid] = tl_offsets[tid - 1] + tl_ub[tid - 1];
    }

    // Allocate padded CRS arrays for C
    int upper_nnz_bound = tl_offsets[n_threads];
    IT *padded_C_col = new IT[upper_nnz_bound];
    bool **used_cols = new bool *[n_threads];
    int *tl_nnz = new int[n_threads];
    IT *C_nnz_per_row = new IT[C_n_rows];

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int tid = 0; tid < n_threads; ++tid) {
            tl_nnz[tid] = 0;
            used_cols[tid] = new bool[B_n_cols];
            for (int i = 0; i < B_n_cols; ++i) {
                used_cols[tid][i] = false;
            }
        }

#pragma omp for schedule(static)
        for (int i = 0; i < C_n_rows; ++i) {
            C_nnz_per_row[i] = 0;
        }
    }

// Padded Gustavson's algorithm (symbolic)
// clang-format off
#pragma omp parallel
    {
        GET_THREAD_ID(int, tid)
        int offset = tl_offsets[tid];
        bool *tl_used_cols = used_cols[tid];

#pragma omp for schedule(static)
        for (int i = 0; i < A_n_rows; ++i) {
            int tl_previous_nnz = tl_nnz[tid];

            for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
                IT left_col = A_col[j];
                for (IT k = B_row_ptr[left_col]; k < B_row_ptr[left_col + 1]; ++k) {
                    IT right_col = B_col[k];
                    if (!tl_used_cols[right_col]) {
                        tl_used_cols[right_col] = true;
                        padded_C_col[offset + tl_nnz[tid]] = right_col;
                        ++tl_nnz[tid];
                        ++C_nnz_per_row[i];
                    }
                }
            }

            // Reset used_cols for next row
            for (int j = tl_previous_nnz; j < tl_nnz[tid]; ++j) {
                tl_used_cols[padded_C_col[offset + j]] = false;
            }

        }
    }
    // clang-format on

    // Build C_row_ptr
    // NOTE: This memory is never deleted by SMAX
    C_row_ptr = new IT[C_n_rows + 1];
    C_row_ptr[0] = 0;
    // NOTE: Does this need to be parallelized?
    for (int i = 0; i < C_n_rows; ++i) {
        C_row_ptr[i + 1] = C_row_ptr[i] + C_nnz_per_row[i];
    }

    // Compute nnz displacement array
    int *C_nnz_displacement = new int[n_threads + 1];
    C_nnz_displacement[0] = 0;
    for (int tid = 1; tid < n_threads + 1; ++tid) {
        C_nnz_displacement[tid] = C_nnz_displacement[tid - 1] + tl_nnz[tid - 1];
    }

    // Allocate the rest of C
    C_nnz = C_row_ptr[C_n_rows];
    C_col = new IT[C_nnz];
    C_val = new VT[C_nnz];

// Compress padded_C_col to C_col
#pragma omp parallel
    {
        GET_THREAD_ID(int, tid)
        int offset = C_nnz_displacement[tid];
        for (int i = 0; i < tl_nnz[tid]; ++i) {
            C_col[C_nnz_displacement[tid] + i] = padded_C_col[offset + i];
        }
    }

    delete[] B_nnz_per_row;
    delete[] tl_ub;
    delete[] tl_offsets;
    delete[] tl_nnz;
    delete[] C_nnz_displacement;
    delete[] C_nnz_per_row;
    delete[] padded_C_col;
    for (int tid = 0; tid < n_threads; ++tid) {
        delete[] used_cols[tid];
    }
    delete[] used_cols;
}

} // namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU
