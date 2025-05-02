#ifndef SPGEMM_NUMERICAL_CPU_IMPL_HPP
#define SPGEMM_NUMERICAL_CPU_IMPL_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spgemm_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {
namespace SPGEMM_CPU {

template <typename IT, typename VT>
inline void
basic_numerical_phase(IT A_n_rows, IT A_n_cols, IT A_nnz, IT *RESTRICT A_col,
                      IT *RESTRICT A_row_ptr, VT *RESTRICT A_val, IT B_n_rows,
                      IT B_n_cols, IT B_nnz, IT *RESTRICT B_col,
                      IT *RESTRICT B_row_ptr, VT *RESTRICT B_val, IT C_n_rows,
                      IT C_n_cols, IT C_nnz, IT *RESTRICT C_col,
                      IT *RESTRICT C_row_ptr, VT *RESTRICT C_val) {

    GET_THREAD_COUNT(IT, num_threads)

    // Prepare thread local dense accumulators
    VT **dense_accumulators = new VT *[num_threads];

    for (IT tid = 0; tid < num_threads; ++tid) {
        dense_accumulators[tid] = new VT[C_n_cols];
    }

#pragma omp parallel
    {
        GET_THREAD_ID(IT, tid)
        for (IT i = 0; i < C_n_cols; ++i) {
            dense_accumulators[tid][i] = (VT)0.0;
        }
    }

// Gustavson's algorithm (numerical)
#pragma omp parallel
    {
        GET_THREAD_ID(IT, tid)
#pragma omp for schedule(static)
        for (IT i = 0; i < A_n_rows; ++i) {
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
                C_val[j] = dense_accumulators[tid][C_col[j]];
                dense_accumulators[tid][C_col[j]] = (VT)0.0;
            }
        }
    }
}

} // namespace SPGEMM_CPU
} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // SPGEMM_NUMERICAL_CPU_IMPL_HPP