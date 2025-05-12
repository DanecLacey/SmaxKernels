#ifndef SMAX_SPTRSV_LVL_CPU_IMPL_HPP
#define SMAX_SPTRSV_LVL_CPU_IMPL_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSV {
namespace SPTRSV_CPU {

template <typename IT, typename VT>
inline void crs_spltrsv_lvl(int A_n_rows, int A_n_cols, int A_nnz,
                            IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                            VT *RESTRICT A_val, VT *RESTRICT x, VT *RESTRICT y,
                            int *lvl_ptr, int n_levels) {

    for (int lvl_idx = 0; lvl_idx < n_levels; ++lvl_idx) {
#pragma omp parallel for
        for (int row = lvl_ptr[lvl_idx]; row < lvl_ptr[lvl_idx + 1]; ++row) {
            VT sum = (VT)0.0;
            VT diag = (VT)0.0;

            for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j) {
                IF_DEBUG(if (A_col[j] < 0 || A_col[j] >= A_n_cols)
                             SpTRSVErrorHandler::col_oob<IT>(A_col[j], j,
                                                             A_n_cols););
                VT val = A_val[j];

                if (A_col[j] < row) {
                    sum += val * x[A_col[j]];
                } else if (A_col[j] == row) {
                    diag = val;
                } else {
                    IF_DEBUG(printf("row: %d, col: %d, val: %f\n", row,
                                    A_col[j], val));
                    IF_DEBUG(SpTRSVErrorHandler::super_diag());
                }
            }

            IF_DEBUG(
                if (abs(diag) < 1e-16) { SpTRSVErrorHandler::zero_diag(); });

            x[row] = (y[row] - sum) / diag;
        }
    }
}

template <typename IT, typename VT>
inline void crs_sputrsv_lvl(int A_n_rows, int A_n_cols, int A_nnz,
                            IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                            VT *RESTRICT A_val, VT *RESTRICT x, VT *RESTRICT y,
                            int *lvl_ptr, int n_levels) {

    for (int lvl_idx = n_levels - 1; lvl_idx >= 0; --lvl_idx) {
#pragma omp parallel for
        // for (int row = lvl_ptr[lvl_idx]; row < lvl_ptr[lvl_idx + 1]; ++row) {
        // DL 12.05.2025 TODO: Does the traversal order within a level
        // even matter?
        for (int row = lvl_ptr[lvl_idx + 1] - 1; row >= lvl_ptr[lvl_idx];
             --row) {
            VT sum = (VT)0.0;
            VT diag = (VT)0.0;

            for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j) {
                IF_DEBUG(if (A_col[j] < 0 || A_col[j] >= A_n_cols)
                             SpTRSVErrorHandler::col_oob<IT>(A_col[j], j,
                                                             A_n_cols););
                VT val = A_val[j];

                if (A_col[j] < row) {
                    sum += val * x[A_col[j]];
                } else if (A_col[j] == row) {
                    diag = val;
                } else {
                    IF_DEBUG(printf("row: %d, col: %d, val: %f\n", row,
                                    A_col[j], val));
                    IF_DEBUG(SpTRSVErrorHandler::super_diag());
                }
            }

            IF_DEBUG(
                if (abs(diag) < 1e-16) { SpTRSVErrorHandler::zero_diag(); });

            x[row] = (y[row] - sum) / diag;
        }
    }
}

} // namespace SPTRSV_CPU
} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_LVL_CPU_IMPL_HPP