#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spgemm_common.hpp"

namespace SMAX::KERNELS::SPGEMM::CPU {

template <typename IT, typename VT>
inline void lvl_seq_traversal(
    Timers *timers, const ULL A_n_rows, const ULL A_n_cols, const ULL A_nnz,
    const IT *SMAX_RESTRICT A_col, const IT *SMAX_RESTRICT A_row_ptr,
    const VT *SMAX_RESTRICT A_val, const ULL B_n_rows, const ULL B_n_cols,
    const ULL B_nnz, const IT *SMAX_RESTRICT B_col,
    const IT *SMAX_RESTRICT B_row_ptr, const VT *SMAX_RESTRICT B_val,
    ULL &C_n_rows, ULL &C_n_cols, ULL &C_nnz, IT *&C_col, IT *&C_row_ptr,
    VT *&C_val, const int *lvl_ptr, const int n_levels) {

    IF_SMAX_TIME(timers->get("Symbolic_Setup")->start());

    SMAX_GET_THREAD_COUNT(ULL, n_threads);

    // Enforce dimensions of C
    C_n_rows = A_n_rows;
    C_n_cols = B_n_cols;
    C_nnz = 0;

    // Allocate C_row_ptr
    // NOTE: This memory is never deleted by SMAX
    C_row_ptr = new IT[C_n_rows + 1];
    C_row_ptr[0] = 0;

    // Count nnz in each row of B
    IT *B_nnz_per_row = new IT[B_n_rows];
    ULL largest_B_row = 0;
    for (ULL i = 0; i < B_n_rows; ++i) {
        B_nnz_per_row[i] = B_row_ptr[i + 1] - B_row_ptr[i];
        if (B_nnz_per_row[i] > largest_B_row)
            largest_B_row = B_nnz_per_row[i];
    }

    ULL *C_nnz_per_row = new ULL[C_n_rows];
    bool *used_cols = new bool[B_n_rows];

    // Allocate used_cols arrays
    for (ULL i = 0; i < B_n_rows; ++i) {
        used_cols[i] = false;
    }

    // Init C_nnz_per_row to 0
    for (ULL i = 0; i < C_n_rows; ++i) {
        C_nnz_per_row[i] = 0;
    }

    // Use maximal row-degree to estimate C_nnz upper bound
    // Fast, but massively overestimates
    ULL C_nnz_mrd = A_nnz * largest_B_row;
    IT *_C_col = new IT[C_nnz_mrd];
    VT *_C_val = new VT[C_nnz_mrd];

    // Prepare dense accumulator (TODO: Bound size)
    VT *dense_accumulator = new VT[C_n_cols];
    for (ULL i = 0; i < C_n_cols; ++i) {
        dense_accumulator[i] = (VT)0.0;
    }

    IF_SMAX_TIME(timers->get("Symbolic_Setup")->stop());
    IF_SMAX_TIME(timers->get("Fused_Gustavson")->start());

    /*
    Traversal order:
                   Read A:  Read B:                  Write C:
    C (L0 symb):   L(0)     L(0), L(1)               L(0)
    C (L0 num):    L(0)     L(0), L(1)               L(0)
    C (L1 symb):   L(1)     L(0), L(1), L(2)         L(1)
    C (L1 num):    L(1)     L(0), L(1), L(2)         L(1)
    C (L2 symb):   L(2)     L(1), L(2), L(3)         L(2)
    C (L2 num):    L(2)     L(1), L(2), L(3)         L(2)
    ...
    C (Ln-2 symb): L(n-2)   L(n-3), L(n-2), L(n-1)   L(n-2)
    C (Ln-2 num):  L(n-2)   L(n-3), L(n-2), L(n-1)   L(n-2)
    C (Ln-1 symb): L(n-1)   L(n-2), L(n-1)           L(n-1)
    C (Ln-1 num):  L(n-1)   L(n-2), L(n-1)           L(n-1)
    */

    // For each level, process the relevant neighbor levels according to the
    // traversal pattern
    // !! Traverse levels of A sequentially
    for (int lvl_idx = 0; lvl_idx < n_levels; ++lvl_idx) {
        // Symbolic part
        for (int row_idx = lvl_ptr[lvl_idx]; row_idx < lvl_ptr[lvl_idx + 1];
             ++row_idx) {
            ULL row_nnz = 0;
            ULL previous_nnz = C_row_ptr[row_idx];
            ULL local_nnz = C_row_ptr[row_idx];

            // For each nonzero in A's row
            for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1]; ++j) {
                IT left_col = A_col[j];

                // It should be guarenteed that these are within L(i-1), L(i),
                // L(i+1)
                IT b_start = B_row_ptr[left_col];
                IT b_end = B_row_ptr[left_col + 1];
                for (IT k = b_start; k < b_end; ++k) {
                    IT right_col = B_col[k];
                    if (!used_cols[right_col]) {
                        used_cols[right_col] = true;
                        _C_col[local_nnz++] = right_col;
                        ++row_nnz;
                    }
                }
            }
            // Reset used_cols for this row
            for (ULL j = previous_nnz; j < local_nnz; ++j) {
                used_cols[_C_col[j]] = false;
            }
            // Store nnz count for this row
            C_nnz += row_nnz;
            C_row_ptr[row_idx + 1] = C_row_ptr[row_idx] + row_nnz;
        }
        // Numerical part
        for (int row_idx = lvl_ptr[lvl_idx]; row_idx < lvl_ptr[lvl_idx + 1];
             ++row_idx) {

            // For each nonzero in A's row
            for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1]; ++j) {
                IT left_col = A_col[j];

                IT b_start = B_row_ptr[left_col];
                IT b_end = B_row_ptr[left_col + 1];
                for (IT k = b_start; k < b_end; ++k) {
                    IT right_col = B_col[k];

                    // Accumulate intermediate results
                    dense_accumulator[right_col] += A_val[j] * B_val[k];
                }
            }
            // Write row-local accumulators to C
            for (IT j = C_row_ptr[row_idx]; j < C_row_ptr[row_idx + 1]; ++j) {

                // clang-format off
                IF_SMAX_DEBUG(
                    if (_C_col[j] < (IT)0 || _C_col[j] >= (IT)C_n_cols)
                        SpGEMMErrorHandler::col_oob<IT>(_C_col[j], j, C_n_cols);
                );
                // clang-format on

                _C_val[j] = dense_accumulator[_C_col[j]];
                dense_accumulator[_C_col[j]] = (VT)0.0;
            }
        }
    }

    IF_SMAX_TIME(timers->get("Fused_Gustavson")->stop());
    IF_SMAX_TIME(timers->get("Compress")->start());

    // NOTE: This memory is never deleted by SMAX
    C_col = new IT[C_nnz];
    C_val = new VT[C_nnz];

    for (ULL i = 0; i < C_nnz; ++i) {
        C_col[i] = _C_col[i];
        C_val[i] = _C_val[i];
    }

    IF_SMAX_TIME(timers->get("Compress")->stop());

    delete[] B_nnz_per_row;
    delete[] C_nnz_per_row;
    delete[] used_cols;
    delete[] _C_col;
    delete[] _C_val;
}

template <typename IT, typename VT>
inline void lvl_parallel_traversal(
    Timers *timers, const ULL A_n_rows, const ULL A_n_cols, const ULL A_nnz,
    const IT *SMAX_RESTRICT A_col, const IT *SMAX_RESTRICT A_row_ptr,
    const VT *SMAX_RESTRICT A_val, const ULL B_n_rows, const ULL B_n_cols,
    const ULL B_nnz, const IT *SMAX_RESTRICT B_col,
    const IT *SMAX_RESTRICT B_row_ptr, const VT *SMAX_RESTRICT B_val,
    ULL &C_n_rows, ULL &C_n_cols, ULL &C_nnz, IT *&C_col, IT *&C_row_ptr,
    VT *&C_val, const int *lvl_ptr, const int n_levels) {

    KernelErrorHandler::not_implemented("lvl_parallel_traversal");
}

template <typename IT, typename VT>
inline void lvl_fused_naive(
    Timers *timers, const ULL A_n_rows, const IT *SMAX_RESTRICT A_col,
    const IT *SMAX_RESTRICT A_row_ptr, const ULL B_n_rows, const ULL B_n_cols,
    const IT *SMAX_RESTRICT B_col, const IT *SMAX_RESTRICT B_row_ptr,
    ULL &C_n_rows, ULL &C_n_cols, ULL &C_nnz, IT *&C_col, IT *&C_row_ptr,
    VT *&C_val, const int *lvl_ptr, const int n_levels) {

    KernelErrorHandler::not_implemented("lvl_fused_naive");

    // ULL *C_nnz_per_row = new ULL[C_n_rows];
    // bool **used_cols = new bool *[n_threads];

    // // clang-format off
    // ULL bounded_size = 0;
    // // TODO: Collect largest level width for bounding used_cols
    // // for (int lvl_idx = 0; lvl_idx < n_levels; ++lvl_idx) {
    // //     int start_lvl = std::max(0, lvl_idx - 1);
    // //     int end_lvl = std::min(n_levels - 1, lvl_idx + 1);

    // //     for (int neighbor_lvl = start_lvl; neighbor_lvl <= end_lvl;
    // ++neighbor_lvl) {
    // //         IT largest_col = 0;
    // //         IT smallest_col = std::numeric_limits<T>::max();
    // //         for (int row_idx = lvl_ptr[neighbor_lvl]; row_idx <
    // lvl_ptr[neighbor_lvl + 1]; ++row_idx) {
    // //             for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx +
    // 1]; ++j) {
    // //                 if(A_col[j] < smallest_col){
    // //                     smallest_col = A_col[j];
    // //                     bounded_size = largest_col - smallest_col;
    // //                 }
    // //                 if(A_col[j] > largest_col){
    // //                     largest_col = A_col[j];
    // //                     bounded_size = largest_col - smallest_col;
    // //                 }
    // //             }
    // //         }
    // //     }
    // // }
    // // clang-format on

    // // Allocate (bounded sized) used_cols arrays
    // bounded_size = B_n_cols;
    // for (ULL tid = 0; tid < n_threads; ++tid) {
    //     used_cols[tid] = new bool[bounded_size];
    //     for (ULL i = 0; i < bounded_size; ++i) {
    //         used_cols[tid][i] = false;
    //     }
    // }

    // // clang-format off
    // for (int lvl_idx = 0; lvl_idx < n_levels; ++lvl_idx) {
    // }
    // // clang-format on
}

} // namespace SMAX::KERNELS::SPGEMM::CPU