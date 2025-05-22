#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmm_common.hpp"

namespace SMAX::KERNELS::SPMM::SPMM_CPU {

template <typename IT, typename VT>
inline void naive_crs_spmm(int A_n_rows, int A_n_cols, IT *RESTRICT A_col,
                           IT *RESTRICT A_row_ptr, VT *RESTRICT A_val,
                           VT *RESTRICT X, VT *RESTRICT Y,
                           int block_vector_size) {

    // clang-format off
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

                IF_SMAX_DEBUG(
                    if (col < 0 || col >= (IT)A_n_cols)
                        SpMMErrorHandler::col_oob<IT>(col, j, A_n_cols);
                );
                IF_SMAX_DEBUG_3(
                    SpMMErrorHandler::print_crs_elem<IT, VT>(
                        A_val[j], col, X[(A_n_rows * vec_idx) + col], j,
                        (A_n_rows * vec_idx) + col);
                );
            }
        }

        for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            Y[row + (vec_idx * A_n_rows)] = tmp[vec_idx];
        }
    }
}
// clang-format on

} // namespace SMAX::KERNELS::SPMM::SPMM_CPU
