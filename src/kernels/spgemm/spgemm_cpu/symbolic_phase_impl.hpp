#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spgemm_common.hpp"

namespace SMAX::KERNELS::SPGEMM::CPU {

template <typename IT, typename VT>
inline void padded_symbolic_phase(Timers *timers, const ULL A_n_rows,
                                  const IT *SMAX_RESTRICT A_col,
                                  const IT *SMAX_RESTRICT A_row_ptr,
                                  const ULL B_n_rows, const ULL B_n_cols,
                                  const IT *SMAX_RESTRICT B_col,
                                  const IT *SMAX_RESTRICT B_row_ptr,
                                  ULL &C_n_rows, ULL &C_n_cols, ULL &C_nnz,
                                  IT *&C_col, IT *&C_row_ptr, VT *&C_val) {

    IF_SMAX_TIME(timers->get("Symbolic_Setup")->start());

    SMAX_GET_THREAD_COUNT(ULL, n_threads);

    // Enforce dimensions of C
    C_n_rows = A_n_rows;
    C_n_cols = B_n_cols;
    C_nnz = 0;

    // Count nnz in each row of B
    IT *B_nnz_per_row = new IT[B_n_rows];
#pragma omp parallel for schedule(static)
    for (ULL i = 0; i < B_n_rows; ++i) {
        B_nnz_per_row[i] = B_row_ptr[i + 1] - B_row_ptr[i];
    }

    // Collect upper-bound offsets for each thread
    ULL *tl_ub = new ULL[n_threads];
    ULL *tl_offsets = new ULL[n_threads + 1];

#pragma omp parallel
    {
        SMAX_GET_THREAD_ID(ULL, tid)
        tl_ub[tid] = 0;

#pragma omp for schedule(static)
        for (ULL i = 0; i < A_n_rows; ++i) {
            ULL local_ub = 0;
            for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
                local_ub += B_nnz_per_row[A_col[j]];
            }
            tl_ub[tid] += local_ub;
        }
    }

    tl_offsets[0] = 0;
    for (ULL tid = 1; tid < n_threads + 1; ++tid) {
        tl_offsets[tid] = tl_offsets[tid - 1] + tl_ub[tid - 1];
    }

    // Allocate padded CRS arrays for C
    ULL upper_nnz_bound = tl_offsets[n_threads];
    IT *SMAX_RESTRICT padded_C_col = new IT[upper_nnz_bound];
    bool **used_cols = new bool *[n_threads];
    ULL *tl_nnz = new ULL[n_threads];
    ULL *C_nnz_per_row = new ULL[C_n_rows];

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (ULL tid = 0; tid < n_threads; ++tid) {
            tl_nnz[tid] = 0;
            used_cols[tid] = new bool[B_n_cols];
            for (ULL i = 0; i < B_n_cols; ++i) {
                used_cols[tid][i] = false;
            }
        }

#pragma omp for schedule(static)
        for (ULL i = 0; i < C_n_rows; ++i) {
            C_nnz_per_row[i] = 0;
        }
    }

    IF_SMAX_TIME(timers->get("Symbolic_Setup")->stop());
    IF_SMAX_TIME(timers->get("Symbolic_Gustavson")->start());

// Padded Gustavson's algorithm (symbolic)
#pragma omp parallel
    {
#ifdef SMAX_USE_LIKWID
        LIKWID_MARKER_START("Symbolic Phase");
#endif
        SMAX_GET_THREAD_ID(ULL, tid)
        ULL offset = tl_offsets[tid];
        bool *tl_used_cols = used_cols[tid];

#pragma omp for schedule(static)
        for (ULL i = 0; i < A_n_rows; ++i) {
            ULL tl_previous_nnz = tl_nnz[tid];
            ULL local_tl_nnz = tl_nnz[tid]; // To help compiler
            ULL local_row_nnz = 0;          // To help compiler

            for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
                IT left_col = A_col[j];
                IT b_start = B_row_ptr[left_col];   // To help compiler
                IT b_end = B_row_ptr[left_col + 1]; // To help compiler
                for (IT k = b_start; k < b_end; ++k) {
                    IT right_col = B_col[k];
                    if (!tl_used_cols[right_col]) {
                        tl_used_cols[right_col] = true;
                        padded_C_col[offset + local_tl_nnz++] = right_col;
                        ++local_row_nnz;
                    }
                }
            }

            // Reset used_cols for next row
            for (ULL j = tl_previous_nnz; j < local_tl_nnz; ++j) {
                tl_used_cols[padded_C_col[offset + j]] = false;
            }

            tl_nnz[tid] = local_tl_nnz;
            C_nnz_per_row[i] = local_row_nnz;
        }
#ifdef SMAX_USE_LIKWID
        LIKWID_MARKER_STOP("Symbolic Phase");
#endif
    }

    IF_SMAX_TIME(timers->get("Symbolic_Gustavson")->stop());
    IF_SMAX_TIME(timers->get("Alloc_C")->start());

    // Build C_row_ptr
    // NOTE: This memory is never deleted by SMAX
    C_row_ptr = new IT[C_n_rows + 1];
    C_row_ptr[0] = 0;
    // NOTE: Does this need to be parallelized?
    for (ULL i = 0; i < C_n_rows; ++i) {
        C_row_ptr[i + 1] = C_row_ptr[i] + C_nnz_per_row[i];
    }

    // Compute nnz displacement array
    ULL *C_nnz_displacement = new ULL[n_threads + 1];
    C_nnz_displacement[0] = 0;
    for (ULL tid = 0; tid < n_threads; ++tid) {
        C_nnz_displacement[tid + 1] = C_nnz_displacement[tid] + tl_nnz[tid];
    }

    // Allocate the rest of C
    C_nnz = C_row_ptr[C_n_rows];
    C_col = new IT[C_nnz];
    C_val = new VT[C_nnz];

    IF_SMAX_TIME(timers->get("Alloc_C")->stop());
    IF_SMAX_TIME(timers->get("Compress")->start());
// Compress padded_C_col to C_col
#pragma omp parallel
    {
        SMAX_GET_THREAD_ID(ULL, tid)
        ULL offset = C_nnz_displacement[tid];
        for (ULL i = 0; i < tl_nnz[tid]; ++i) {
            C_col[offset + i] = padded_C_col[tl_offsets[tid] + i];
        }
    }
    IF_SMAX_TIME(timers->get("Compress")->stop());

    delete[] B_nnz_per_row;
    delete[] tl_ub;
    delete[] tl_offsets;
    delete[] tl_nnz;
    delete[] C_nnz_displacement;
    delete[] C_nnz_per_row;
    delete[] padded_C_col;
    for (ULL tid = 0; tid < n_threads; ++tid) {
        delete[] used_cols[tid];
    }
    delete[] used_cols;
}

} // namespace SMAX::KERNELS::SPGEMM::CPU
