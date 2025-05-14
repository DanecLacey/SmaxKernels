#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsm_common.hpp"

namespace SMAX::KERNELS::SPTRSM::SPTRSM_CPU {

template <typename IT, typename VT>
inline void naive_crs_spltrsm(int A_n_rows, int A_n_cols, int A_nnz,
                              IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                              VT *RESTRICT A_val, VT *RESTRICT D_val,
                              VT *RESTRICT X, VT *RESTRICT Y,
                              int block_vector_size) {

    for (int row = 0; row < A_n_rows; ++row) {
        for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            VT sum = (VT)0.0;
            for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1] - 1; ++j) {
                IT col = A_col[j];

                IF_DEBUG(
                    if (col < 0 || col >= A_n_cols)
                        SpTRSMErrorHandler::col_oob<IT>(col, j, A_n_cols););

                IF_DEBUG(if (col > row) SpTRSMErrorHandler::super_diag(
                             row, col, A_val[j]););

                sum += A_val[j] * X[(A_n_rows * vec_idx) + col];
            }

            IF_DEBUG(if (std::abs(D_val[row]) < 1e-16) {
                SpTRSMErrorHandler::zero_diag(row);
            });

            X[row + (A_n_rows * vec_idx)] =
                (Y[row + (A_n_rows * vec_idx)] - sum) / D_val[row];
        }
    }
}

template <typename IT, typename VT>
inline void naive_crs_sputrsm(int A_n_rows, int A_n_cols, int A_nnz,
                              IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                              VT *RESTRICT A_val, VT *RESTRICT D_val,
                              VT *RESTRICT X, VT *RESTRICT Y,
                              int block_vector_size) {

    for (int row = A_n_rows - 1; row >= 0; --row) {
        for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
            VT sum = (VT)0.0;
            for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1] - 1; ++j) {
                IT col = A_col[j];

                IF_DEBUG(
                    if (col < 0 || col >= A_n_cols)
                        SpTRSMErrorHandler::col_oob<IT>(col, j, A_n_cols););

                IF_DEBUG(if (col > row) SpTRSMErrorHandler::super_diag(
                             row, col, A_val[j]););

                sum += A_val[j] * X[(A_n_rows * vec_idx) + col];
            }

            IF_DEBUG(if (std::abs(D_val[row]) < 1e-16) {
                SpTRSMErrorHandler::zero_diag(row);
            });

            X[row + (A_n_rows * vec_idx)] =
                (Y[row + (A_n_rows * vec_idx)] - sum) / D_val[row];
        }
    }
}

} // namespace SMAX::KERNELS::SPTRSM::SPTRSM_CPU
