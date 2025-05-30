#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsm_common.hpp"

namespace SMAX::KERNELS::SPTRSM::SPTRSM_CPU {

template <bool Lower, typename IT, typename VT>
inline void naive_crs_col_maj_sptrsm(int A_n_rows, int A_n_cols,
                                     IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                                     VT *RESTRICT A_val, VT *RESTRICT D_val,
                                     VT *RESTRICT X, VT *RESTRICT Y,
                                     int block_vector_size) {

    // clang-format off
    int row_start, row_end, row_step;
    if constexpr (Lower) {
        row_start = 0; row_end = A_n_rows; row_step = 1;
    } else {
        row_start = A_n_rows - 1; row_end = -1; row_step = -1;
    }

    for (int row = row_start; row != row_end; row += row_step) {
        for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            VT sum = (VT)0.0;
            for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1] - 1; ++j) {
                IT col = A_col[j];

                IF_SMAX_DEBUG(
                    if (col < (IT)0 || col >= (IT)A_n_cols)
                        SpTRSMErrorHandler::col_oob<IT>(col, j, A_n_cols)
                    if constexpr (Lower){
                        if (col > (IT)row)
                            SpTRSMErrorHandler::super_diag(row, col, A_val[j]);
                    }
                    else{
                        if (col < (IT)row)
                            SpTRSMErrorHandler::sub_diag(row, col, A_val[j]);
                    }
                );

                sum += A_val[j] * X[(A_n_rows * vec_idx) + col];
            }

            IF_SMAX_DEBUG(
                if (std::abs(D_val[row]) < 1e-16)
                    SpTRSMErrorHandler::zero_diag(row)
            );

            X[row + (A_n_rows * vec_idx)] =
                (Y[row + (A_n_rows * vec_idx)] - sum) / D_val[row];
        }
    }
}

template <bool Lower, typename IT, typename VT>
inline void naive_crs_row_maj_sptrsm(int A_n_rows, int A_n_cols,
                                     IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                                     VT *RESTRICT A_val, VT *RESTRICT D_val,
                                     VT *RESTRICT X, VT *RESTRICT Y,
                                     int block_vector_size) {

    // clang-format off
    int row_start, row_end, row_step;
    if constexpr (Lower) {
        row_start = 0; row_end = A_n_rows; row_step = 1;
    } else {
        row_start = A_n_rows - 1; row_end = -1; row_step = -1;
    }

    for (int row = row_start; row != row_end; row += row_step) {
        for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            VT sum = (VT)0.0;
            for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1] - 1; ++j) {
                IT col = A_col[j];

                IF_SMAX_DEBUG(
                    if (col < (IT)0 || col >= (IT)A_n_cols)
                        SpTRSMErrorHandler::col_oob<IT>(col, j, A_n_cols)
                    if constexpr (Lower){
                        if (col > (IT)row)
                            SpTRSMErrorHandler::super_diag(row, col, A_val[j]);
                    }
                    else{
                        if (col < (IT)row)
                            SpTRSMErrorHandler::sub_diag(row, col, A_val[j]);
                    }
                );

                sum += A_val[j] * X[col * block_vector_size + vec_idx];
            }

            IF_SMAX_DEBUG(
                if (std::abs(D_val[row]) < 1e-16)
                    SpTRSMErrorHandler::zero_diag(row)
            );

            X[row * block_vector_size + vec_idx] =
                (Y[row * block_vector_size + vec_idx] - sum) / D_val[row];
        }
    }
    // clang-format on
}

} // namespace SMAX::KERNELS::SPTRSM::SPTRSM_CPU
