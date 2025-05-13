#ifndef SMAX_SPMM_CPU_IMPL_HPP
#define SMAX_SPMM_CPU_IMPL_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmm_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMM {
namespace SPMM_CPU {

template <typename IT, typename VT>
inline void naive_crs_spmm(int A_n_rows, int A_n_cols, int A_nnz,
                           IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                           VT *RESTRICT A_val, VT *RESTRICT X, VT *RESTRICT Y,
                           int block_vector_size) {

// Assuming colwise layout for now
#pragma omp parallel for schedule(static)
    for (int row = 0; row < A_n_rows; ++row) {
        VT tmp[block_vector_size];

        for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            tmp[vec_idx] = VT{};
        }

        for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j) {
            IT col = A_col[j];
#pragma omp simd
            for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
                tmp[vec_idx] += A_val[j] * X[(A_n_rows * vec_idx) + col];

                IF_DEBUG(
#if DEBUG_LEVEL == 3
                    printf("A_val[%d] = %f\n", j, A_val[j]);
                    printf("A_col[%d] = %d\n", j, col);
                    printf("(A_n_rows * vec_idx) + A_col[%d] = %d\n", j,
                           (A_n_rows * vec_idx) + col);
                    printf("X[(A_n_rows * vec_idx) + A_col[%d]] = %f\n", j,
                           X[(A_n_rows * vec_idx) + col]);
#endif
                    if (col < 0 || col >= (IT)A_n_cols)
                        SpMMErrorHandler::col_oob<IT>(col, j, A_n_cols););
            }
        }

        for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            Y[row + (vec_idx * A_n_rows)] = tmp[vec_idx];
        }
    }
}

} // namespace SPMM_CPU
} // namespace SPMM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMM_CPU_IMPL_HPP