#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX::KERNELS::SPTRSV::CPU {

template <bool Lower, typename IT, typename VT>
inline void
naive_crs_sptrsv(const ULL n_rows, const ULL n_cols,
                 const IT *SMAX_RESTRICT col, const IT *SMAX_RESTRICT row_ptr,
                 const VT *SMAX_RESTRICT val, const VT *SMAX_RESTRICT D_val,
                 VT *SMAX_RESTRICT x, const VT *SMAX_RESTRICT y) {

    // clang-format off
    long long int row_start, row_end, row_step;
    if constexpr (Lower) {
        row_start = 0; row_end = n_rows; row_step = 1;
    } else {
        row_start = n_rows - 1; row_end = -1; row_step = -1;
    }
    
    for (long long int row = row_start; row != row_end; row += row_step) {
        VT sum = (VT)0.0;

        // NOTE: we assume the diagonal was sorted to the end of the row
        for (IT j = row_ptr[row]; j < row_ptr[row + 1] - 1; ++j) {

            IF_SMAX_DEBUG(
                if (col[j] < (IT)0 || col[j] >= (IT)n_cols)
                    SpTRSVErrorHandler::col_oob<IT>(col[j], j, n_cols);
            );
            IF_SMAX_DEBUG(
                if constexpr (Lower){
                    if (col[j] > (IT)row)
                        SpTRSVErrorHandler::super_diag(row, col[j], val[j]);
                }
                else{
                    if (col[j] < (IT)row)
                        SpTRSVErrorHandler::sub_diag(row, col[j], val[j]);
                }
            );

            sum += val[j] * x[col[j]];
        }

        IF_SMAX_DEBUG(
            if (std::abs(D_val[row]) < 1e-16)
                SpTRSVErrorHandler::zero_diag(row);
        );

        x[row] = (y[row] - sum) / D_val[row];
    }
}

} // namespace SMAX::KERNELS::SPTRSV::CPU
