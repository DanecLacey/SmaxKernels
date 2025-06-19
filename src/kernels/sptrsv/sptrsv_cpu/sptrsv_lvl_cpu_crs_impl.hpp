#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX::KERNELS::SPTRSV::CPU {

// TODO: Will need to update lvl_ptr to ULL eventually

template <bool Lower, typename IT, typename VT>
inline void crs_sptrsv_lvl(const int n_levels, const ULL n_cols,
                           const IT *RESTRICT col, const IT *RESTRICT row_ptr,
                           const VT *RESTRICT val, const VT *RESTRICT D,
                           VT *RESTRICT x, const VT *RESTRICT y,
                           const int *lvl_ptr) {

    // DL 30.05.25 NOTE: Cannot do too much DRY due to OpenMP
    // clang-format off
    if constexpr (Lower) {
        for (int lvl_idx = 0; lvl_idx < n_levels; ++lvl_idx) {
#pragma omp parallel for
            for (int row_idx = lvl_ptr[lvl_idx]; row_idx < lvl_ptr[lvl_idx + 1]; ++row_idx) {
                VT sum = (VT)0.0;
                IT row_start = row_ptr[row_idx]; // To help compiler
                IT row_end   = row_ptr[row_idx + 1] - 1; // To help compiler

                #pragma omp simd reduction(+:sum)
                for (IT j = row_start; j < row_end; ++j) {

                    IF_SMAX_DEBUG(
                        if (col[j] < (IT)0 || col[j] >= (IT)n_cols)
                            SpTRSVErrorHandler::col_oob<IT>(col[j], j, n_cols);
                        if (col[j] > (IT)row_idx)
                            SpTRSVErrorHandler::super_diag(row_idx, col[j], val[j]);
                    );

                    sum += val[j] * x[col[j]];
                }

                IF_SMAX_DEBUG(
                    if (std::abs(D[row_idx]) < 1e-16)
                        SpTRSVErrorHandler::zero_diag(row_idx);
                );

                x[row_idx] = (y[row_idx] - sum) / D[row_idx];
            }
        }
    } else {
        for (int lvl_idx = n_levels - 1; lvl_idx >= 0; --lvl_idx) {
    #pragma omp parallel for
            // DL 12.05.2025 NOTE: Traversal order within a level does not matter.
            for (int row_idx = lvl_ptr[lvl_idx]; row_idx < lvl_ptr[lvl_idx + 1]; ++row_idx) {
                VT sum = (VT)0.0;
                IT row_start = row_ptr[row_idx]; // To help compiler
                IT row_end   = row_ptr[row_idx + 1] - 1; // To help compiler

                #pragma omp simd reduction(+:sum)
                for (IT j = row_start; j < row_end; ++j) {

                    IF_SMAX_DEBUG(
                        if (col[j] < (IT)0 || col[j] >= (IT)n_cols)
                            SpTRSVErrorHandler::col_oob<IT>(col[j], j, n_cols);
                        if (col[j] < (IT)row_idx)
                            SpTRSVErrorHandler::sub_diag(row_idx, col[j], val[j]);
                    );

                    sum += val[j] * x[col[j]];
                }

                IF_SMAX_DEBUG(
                    if (std::abs(D[row_idx]) < 1e-16)
                        SpTRSVErrorHandler::zero_diag(row_idx);
                );

                x[row_idx] = (y[row_idx] - sum) / D[row_idx];
            }
        }
    }
    // clang-format on
}

} // namespace SMAX::KERNELS::SPTRSV::CPU
