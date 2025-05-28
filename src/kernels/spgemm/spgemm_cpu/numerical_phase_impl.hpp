#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spgemm_common.hpp"

namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU {

template <typename IT, typename VT>
inline void basic_numerical_phase(Timers *timers, int A_n_rows,
                                  IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                                  VT *RESTRICT A_val, IT *RESTRICT B_col,
                                  IT *RESTRICT B_row_ptr, VT *RESTRICT B_val,
                                  int C_n_cols, IT *RESTRICT C_col,
                                  IT *RESTRICT C_row_ptr, VT *RESTRICT C_val) {

    IF_SMAX_TIME(timers->get("Numerical_Setup")->start());

    GET_THREAD_COUNT(int, n_threads)

    // Prepare thread local dense accumulators
    VT **dense_accumulators = new VT *[n_threads];

    for (int tid = 0; tid < n_threads; ++tid) {
        dense_accumulators[tid] = new VT[C_n_cols];
    }

#pragma omp parallel
    {
        GET_THREAD_ID(int, tid)
        for (int i = 0; i < C_n_cols; ++i) {
            dense_accumulators[tid][i] = (VT)0.0;
        }
    }

    IF_SMAX_TIME(timers->get("Numerical_Setup")->stop());
    IF_SMAX_TIME(timers->get("Numerical_Gustavson")->start());

// Gustavson's algorithm (numerical)
#pragma omp parallel
    {
        GET_THREAD_ID(int, tid)
#pragma omp for schedule(static)
        for (int i = 0; i < A_n_rows; ++i) {
            for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
                IT left_col = A_col[j];
                for (IT k = B_row_ptr[left_col]; k < B_row_ptr[left_col + 1];
                     ++k) {
                    IT right_col = B_col[k];

                    // Accumulate intermediate results
                    dense_accumulators[tid][right_col] += A_val[j] * B_val[k];
                }
            }
            // Write row-local accumulators to C
            for (IT j = C_row_ptr[i]; j < C_row_ptr[i + 1]; ++j) {

                // clang-format off
                IF_SMAX_DEBUG(
                    if (C_col[j] < (IT)0 || C_col[j] >= (IT)C_n_cols)
                        SpGEMMErrorHandler::col_oob<IT>(C_col[j], j, C_n_cols);
                );
                // clang-format on

                C_val[j] = dense_accumulators[tid][C_col[j]];
                dense_accumulators[tid][C_col[j]] = (VT)0.0;
            }
        }
    }
    IF_SMAX_TIME(timers->get("Numerical_Gustavson")->stop());

    for (int tid = 0; tid < n_threads; ++tid) {
        delete[] dense_accumulators[tid];
    }
    delete[] dense_accumulators;
}

} // namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU
