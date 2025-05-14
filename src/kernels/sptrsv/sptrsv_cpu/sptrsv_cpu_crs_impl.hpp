#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX::KERNELS::SPTRSV::SPTRSV_CPU {

// TODO: JH

// // https://icl.utk.edu/files/publications/2018/icl-utk-1067-2018.pdf
// // https://www.nrel.gov/docs/fy22osti/80263.pdf
// void spltsv_2stage(

// ){
//     // TODO
// }

// // Saad: Iterative Methods for Sparse Linear Systems (ch 12.4.3)
// void spltsv_mc(

// ){
//     // TODO
// }

template <typename IT, typename VT>
inline void naive_crs_spltrsv(int A_n_rows, int A_n_cols, int A_nnz,
                              IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                              VT *RESTRICT A_val, VT *RESTRICT D_val,
                              VT *RESTRICT x, VT *RESTRICT y) {
    for (int row_idx = 0; row_idx < A_n_rows; ++row_idx) {
        VT sum = (VT)0.0;

        // NOTE: we assume the diagonal was sorted to the end of the row
        for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1] - 1; ++j) {
            IT col = A_col[j];

            IF_DEBUG(if (col < 0 || col >= A_n_cols)
                         SpTRSVErrorHandler::col_oob<IT>(col, j, A_n_cols););

            sum += A_val[j] * x[col];

            IF_DEBUG(if (col > row_idx) SpTRSVErrorHandler::super_diag(
                         row_idx, col, A_val[j]););
        }

        IF_DEBUG(if (std::abs(D_val[row_idx]) < 1e-16) {
            SpTRSVErrorHandler::zero_diag(row_idx);
        });

        x[row_idx] = (y[row_idx] - sum) / D_val[row_idx];
    }
}

template <typename IT, typename VT>
inline void naive_crs_sputrsv(int A_n_rows, int A_n_cols, int A_nnz,
                              IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                              VT *RESTRICT A_val, VT *RESTRICT D_val,
                              VT *RESTRICT x, VT *RESTRICT y) {
    for (int row_idx = A_n_rows - 1; row_idx >= 0; --row_idx) {
        VT sum = (VT)0.0;

        for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1] - 1; ++j) {
            IT col = A_col[j];

            IF_DEBUG(if (col < 0 || col >= A_n_cols)
                         SpTRSVErrorHandler::col_oob<IT>(col, j, A_n_cols););

            sum += A_val[j] * x[col];

            IF_DEBUG(if (col < row_idx)
                         SpTRSVErrorHandler::sub_diag(row_idx, col, A_val[j]););
        }

        IF_DEBUG(if (D_val[row_idx] < 1e-16) {
            SpTRSVErrorHandler::zero_diag(row_idx);
        });

        x[row_idx] = (y[row_idx] - sum) / D_val[row_idx];
    }
}

} // namespace SMAX::KERNELS::SPTRSV::SPTRSV_CPU
