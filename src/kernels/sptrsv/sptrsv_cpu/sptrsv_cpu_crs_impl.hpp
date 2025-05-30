#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX::KERNELS::SPTRSV::SPTRSV_CPU {

template <bool Lower, typename IT, typename VT>
inline void naive_crs_sptrsv(int A_n_rows, int A_n_cols, IT *RESTRICT A_col,
                             IT *RESTRICT A_row_ptr, VT *RESTRICT A_val,
                             VT *RESTRICT D_val, VT *RESTRICT x,
                             VT *RESTRICT y) {

    // clang-format off
    int row_start, row_end, row_step;
    if constexpr (Lower) {
        row_start = 0; row_end = A_n_rows; row_step = 1;
    } else {
        row_start = A_n_rows - 1; row_end = -1; row_step = -1;
    }
    
    for (int row = row_start; row != row_end; row += row_step) {
        VT sum = (VT)0.0;

        // NOTE: we assume the diagonal was sorted to the end of the row
        for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1] - 1; ++j) {
            IT col = A_col[j];

            IF_SMAX_DEBUG(
                if (col < (IT)0 || col >= (IT)A_n_cols)
                    SpTRSVErrorHandler::col_oob<IT>(col, j, A_n_cols);
            );
            IF_SMAX_DEBUG(
                if constexpr (Lower){
                    if (col > (IT)row)
                        SpTRSVErrorHandler::super_diag(row, col, A_val[j]);
                }
                else{
                    if (col < (IT)row)
                        SpTRSVErrorHandler::sub_diag(row, col, A_val[j]);
                }
            );

            sum += A_val[j] * x[col];
        }

        IF_SMAX_DEBUG(
            if (std::abs(D_val[row]) < 1e-16)
                SpTRSVErrorHandler::zero_diag(row);
        );

        x[row] = (y[row] - sum) / D_val[row];
    }
}

} // namespace SMAX::KERNELS::SPTRSV::SPTRSV_CPU
