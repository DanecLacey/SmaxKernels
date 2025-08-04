#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spgemm_common.hpp"

namespace SMAX::KERNELS::SPGEMM::CPU {

template <typename IT, typename VT>
inline void basic_numerical_phase(
    Timers *timers, const ULL A_n_rows, const IT *SMAX_RESTRICT A_col,
    const IT *SMAX_RESTRICT A_row_ptr, const VT *SMAX_RESTRICT A_val,
    const IT *SMAX_RESTRICT B_col, const IT *SMAX_RESTRICT B_row_ptr,
    const VT *SMAX_RESTRICT B_val, const ULL C_n_cols,
    const IT *SMAX_RESTRICT C_col, const IT *SMAX_RESTRICT C_row_ptr,
    VT *SMAX_RESTRICT C_val) {

    IF_SMAX_TIME(timers->get("Numerical_Setup")->start());

    SMAX_GET_THREAD_COUNT(ULL, n_threads)

    // Prepare thread local dense accumulators
    VT **dense_accumulators = new VT *[n_threads];

    for (ULL tid = 0; tid < n_threads; ++tid) {
        dense_accumulators[tid] = new VT[C_n_cols];
    }

#pragma omp parallel
    {
        SMAX_GET_THREAD_ID(ULL, tid)
        for (ULL i = 0; i < C_n_cols; ++i) {
            dense_accumulators[tid][i] = (VT)0.0;
        }
    }

    ULL *tl_n_mults = new ULL[n_threads];

    IF_SMAX_TIME(timers->get("Numerical_Setup")->stop());
    IF_SMAX_TIME(timers->get("Numerical_Gustavson")->start());

// Gustavson's algorithm (numerical)
#pragma omp parallel
    {
        SMAX_GET_THREAD_ID(ULL, tid)
        IF_SMAX_DEBUG(tl_n_mults[tid] = 0);
#pragma omp for schedule(static)
        for (ULL i = 0; i < A_n_rows; ++i) {
            for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
                IT left_col = A_col[j];
                for (IT k = B_row_ptr[left_col]; k < B_row_ptr[left_col + 1];
                     ++k) {
                    IT right_col = B_col[k];

                    // Accumulate intermediate results
                    dense_accumulators[tid][right_col] += A_val[j] * B_val[k];
                    IF_SMAX_DEBUG(++tl_n_mults[tid]);
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
    IF_SMAX_DEBUG(ULL n_mults = 0; for (ULL tid = 0; tid < n_threads; ++tid) {
        n_mults += tl_n_mults[tid];
    } ErrorHandler::log("n_mults = %llu", n_mults););

    for (ULL tid = 0; tid < n_threads; ++tid) {
        delete[] dense_accumulators[tid];
    }
    delete[] dense_accumulators;
    delete[] tl_n_mults;
}

} // namespace SMAX::KERNELS::SPGEMM::CPU
