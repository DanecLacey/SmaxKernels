#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmm_common.hpp"

namespace SMAX::KERNELS::SPMM::CPU {

template <typename IT, typename VT>
inline void
naive_crs_spmm_col_maj(const ULL n_rows, const ULL n_cols,
                       const IT *RESTRICT col, const IT *RESTRICT row_ptr,
                       const VT *RESTRICT val, const VT *RESTRICT X,
                       VT *RESTRICT Y, const ULL block_vector_size) {

    // clang-format off
#pragma omp parallel for schedule(static)
    for (ULL row = 0; row < n_rows; ++row) {
        VT tmp[block_vector_size];

        for (ULL vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            tmp[vec_idx] = VT{};
        }

        for (IT j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
#pragma omp simd
            for (ULL vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
                tmp[vec_idx] += val[j] * X[(n_rows * vec_idx) + col[j]];

                IF_SMAX_DEBUG(
                    if (col[j] < 0 || col[j] >= (IT)n_cols)
                        SpMMErrorHandler::col_oob<IT>(col[j], j, n_cols);
                );
                IF_SMAX_DEBUG_3(
                    SpMMErrorHandler::print_crs_elem<IT, VT>(
                        val[j], col[j], X[(n_rows * vec_idx) + col[j]], j,
                        (n_rows * vec_idx) + col[j]);
                );
            }
        }

        for (ULL vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            Y[row + (vec_idx * n_rows)] = tmp[vec_idx];
        }
    }
}
// clang-format on

template <typename IT, typename VT>
inline void
naive_crs_spmm_row_maj(const ULL n_rows, const ULL n_cols,
                       const IT *RESTRICT col, const IT *RESTRICT row_ptr,
                       const VT *RESTRICT val, const VT *RESTRICT X,
                       VT *RESTRICT Y, const ULL block_vector_size) {

    // clang-format off
#pragma omp parallel for schedule(static)
    for (ULL row = 0; row < n_rows; ++row) {
        VT tmp[block_vector_size];

        for (ULL vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            tmp[vec_idx] = VT{};
        }

        for (IT j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
#pragma omp simd
            for (ULL vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {

                tmp[vec_idx] += val[j] * X[col[j] * block_vector_size + vec_idx];

                IF_SMAX_DEBUG(
                    if (col[j] < 0 || col[j] >= (IT)n_cols)
                        SpMMErrorHandler::col_oob<IT>(col[j], j, n_cols);
                );
                IF_SMAX_DEBUG_3(
                    SpMMErrorHandler::print_crs_elem<IT, VT>(
                        val[j], col[j], X[(n_rows * vec_idx) + col[j]], j,
                        (n_rows * vec_idx) + col[j]);
                );
            }
        }

        for (ULL vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            Y[row * block_vector_size + vec_idx] = tmp[vec_idx];
        }
    }
}
// clang-format on

} // namespace SMAX::KERNELS::SPMM::CPU
