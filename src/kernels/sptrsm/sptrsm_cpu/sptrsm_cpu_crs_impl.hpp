#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsm_common.hpp"

namespace SMAX::KERNELS::SPTRSM::CPU {

template <bool Lower, typename IT, typename VT>
inline void naive_crs_col_maj_sptrsm(
    const ULL n_rows, const ULL n_cols, const IT *SMAX_RESTRICT col,
    const IT *SMAX_RESTRICT row_ptr, const VT *SMAX_RESTRICT val,
    const VT *SMAX_RESTRICT D, VT *SMAX_RESTRICT X, const VT *SMAX_RESTRICT Y,
    const ULL block_vector_size) {

    // clang-format off
    long long int row_start, row_end, row_step;
    if constexpr (Lower) {
        row_start = 0; row_end = n_rows; row_step = 1;
    } else {
        row_start = n_rows - 1; row_end = -1; row_step = -1;
    }

    for (long long int row = row_start; row != row_end; row += row_step) {
        for (ULL vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            VT sum = (VT)0.0;
            for (IT j = row_ptr[row]; j < row_ptr[row + 1] - 1; ++j) {

                IF_SMAX_DEBUG(
                    if (col[j] < (IT)0 || col[j] >= (IT)n_cols)
                        SpTRSMErrorHandler::col_oob<IT>(col[j], j, n_cols);
                    if constexpr (Lower){
                        if (col[j] > (IT)row)
                            SpTRSMErrorHandler::super_diag(row, col[j], val[j]);
                    }
                    else{
                        if (col[j] < (IT)row)
                            SpTRSMErrorHandler::sub_diag(row, col[j], val[j]);
                    }
                );

                sum += val[j] * X[(n_rows * vec_idx) + col[j]];
            }

            IF_SMAX_DEBUG(
                if (std::abs(D[row]) < 1e-16)
                    SpTRSMErrorHandler::zero_diag(row)
            );

            X[row + (n_rows * vec_idx)] =
                (Y[row + (n_rows * vec_idx)] - sum) / D[row];
        }
    }
}

template <bool Lower, typename IT, typename VT>
inline void naive_crs_row_maj_sptrsm(const ULL n_rows, const ULL n_cols,
                                     const IT *SMAX_RESTRICT col, const IT *SMAX_RESTRICT row_ptr,
                                     const VT *SMAX_RESTRICT val, const VT *SMAX_RESTRICT D,
                                     VT *SMAX_RESTRICT X, const VT *SMAX_RESTRICT Y,
                                     const ULL block_vector_size) {

    // clang-format off
    long long int row_start, row_end, row_step;
    if constexpr (Lower) {
        row_start = 0; row_end = n_rows; row_step = 1;
    } else {
        row_start = n_rows - 1; row_end = -1; row_step = -1;
    }

    for (long long int row = row_start; row != row_end; row += row_step) {
        for (ULL vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            VT sum = (VT)0.0;
            for (IT j = row_ptr[row]; j < row_ptr[row + 1] - 1; ++j) {

                IF_SMAX_DEBUG(
                    if (col[j] < (IT)0 || col[j] >= (IT)n_cols)
                        SpTRSMErrorHandler::col_oob<IT>(col[j], j, n_cols);
                    if constexpr (Lower){
                        if (col[j] > (IT)row)
                            SpTRSMErrorHandler::super_diag(row, col[j], val[j]);
                    }
                    else{
                        if (col[j] < (IT)row)
                            SpTRSMErrorHandler::sub_diag(row, col[j], val[j]);
                    }
                );

                sum += val[j] * X[col[j] * block_vector_size + vec_idx];
            }

            IF_SMAX_DEBUG(
                if (std::abs(D[row]) < 1e-16)
                    SpTRSMErrorHandler::zero_diag(row)
            );

            X[row * block_vector_size + vec_idx] =
                (Y[row * block_vector_size + vec_idx] - sum) / D[row];
        }
    }
    // clang-format on
}

} // namespace SMAX::KERNELS::SPTRSM::CPU
