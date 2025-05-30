#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX::KERNELS::SPTRSV::SPTRSV_CPU {

template <bool Lower, typename IT, typename VT>
inline void crs_sptrsv_lvl(int n_levels, int A_n_cols, IT *RESTRICT A_col,
                           IT *RESTRICT A_row_ptr, VT *RESTRICT A_val,
                           VT *RESTRICT D_val, VT *RESTRICT x, VT *RESTRICT y,
                           int *lvl_ptr) {

    // DL 30.05.25 NOTE: Cannot do too much DRY due to OpenMP
    // clang-format off
    if constexpr (Lower) {
        for (int lvl_idx = 0; lvl_idx < n_levels; ++lvl_idx) {
#pragma omp parallel for
            for (int row_idx = lvl_ptr[lvl_idx]; row_idx < lvl_ptr[lvl_idx + 1]; ++row_idx) {
                VT sum = (VT)0.0;

                for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1] - 1; ++j) {
                    IT col = A_col[j];
                    
                    IF_SMAX_DEBUG(
                        if (col < (IT)0 || col >= (IT)A_n_cols)
                            SpTRSVErrorHandler::col_oob<IT>(col, j, A_n_cols);
                        if (col > (IT)row_idx)
                            SpTRSVErrorHandler::super_diag(row_idx, col, A_val[j]);
                    );

                    sum += A_val[j] * x[col];
                }

                IF_SMAX_DEBUG(
                    if (std::abs(D_val[row_idx]) < 1e-16)
                        SpTRSVErrorHandler::zero_diag(row_idx);
                );

                x[row_idx] = (y[row_idx] - sum) / D_val[row_idx];
            }
        }
    } else {
        for (int lvl_idx = n_levels - 1; lvl_idx >= 0; --lvl_idx) {
    #pragma omp parallel for
            // DL 12.05.2025 NOTE: Traversal order within a level does not matter.
            for (int row_idx = lvl_ptr[lvl_idx]; row_idx < lvl_ptr[lvl_idx + 1]; ++row_idx) {
                VT sum = (VT)0.0;

                for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1] - 1; ++j) {
                    IT col = A_col[j];

                    IF_SMAX_DEBUG(
                        if (col < (IT)0 || col >= (IT)A_n_cols)
                            SpTRSVErrorHandler::col_oob<IT>(col, j, A_n_cols);
                        if (col < (IT)row_idx)
                            SpTRSVErrorHandler::sub_diag(row_idx, col, A_val[j]);
                    );

                    sum += A_val[j] * x[col];
                }

                IF_SMAX_DEBUG(
                    if (std::abs(D_val[row_idx]) < 1e-16)
                        SpTRSVErrorHandler::zero_diag(row_idx);
                );

                x[row_idx] = (y[row_idx] - sum) / D_val[row_idx];
            }
        }
    }
    // clang-format on
}

} // namespace SMAX::KERNELS::SPTRSV::SPTRSV_CPU
