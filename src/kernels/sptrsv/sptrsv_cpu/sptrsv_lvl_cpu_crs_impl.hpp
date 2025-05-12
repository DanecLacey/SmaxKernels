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
inline void
crs_spltrsv_lvl(int A_n_rows, int A_n_cols, int A_nnz, IT *RESTRICT A_col,
                IT *RESTRICT A_row_ptr, VT *RESTRICT A_val, VT *RESTRICT D_val,
                VT *RESTRICT x, VT *RESTRICT y, int *lvl_ptr, int n_levels) {

    for (int lvl_idx = 0; lvl_idx < n_levels; ++lvl_idx) {
#pragma omp parallel for
        for (int row_idx = lvl_ptr[lvl_idx]; row_idx < lvl_ptr[lvl_idx + 1];
             ++row_idx) {
            VT sum = (VT)0.0;

            for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1]; ++j) {
                IT col = A_col[j];

                IF_DEBUG(
                    if (col < 0 || col >= A_n_cols)
                        SpTRSVErrorHandler::col_oob<IT>(col, j, A_n_cols););

                sum += A_val[j] * x[col];

                IF_DEBUG(if (col > row_idx) SpTRSVErrorHandler::super_diag(
                             row_idx, col, A_val[j]););
            }

            IF_DEBUG(if (D_val[row_idx] < 1e-16) {
                SpTRSVErrorHandler::zero_diag();
            });

            x[row_idx] = (y[row_idx] - sum) / D_val[row_idx];
        }
    }
}

template <typename IT, typename VT>
inline void
crs_sputrsv_lvl(int A_n_rows, int A_n_cols, int A_nnz, IT *RESTRICT A_col,
                IT *RESTRICT A_row_ptr, VT *RESTRICT A_val, VT *RESTRICT D_val,
                VT *RESTRICT x, VT *RESTRICT y, int *lvl_ptr, int n_levels) {

    for (int lvl_idx = n_levels - 1; lvl_idx >= 0; --lvl_idx) {
#pragma omp parallel for
        // for (int row_idx = lvl_ptr[lvl_idx]; row_idx < lvl_ptr[lvl_idx + 1];
        // ++row_idx) { DL 12.05.2025 TODO: Does the traversal order within a
        // level even matter?
        for (int row_idx = lvl_ptr[lvl_idx + 1] - 1;
             row_idx >= lvl_ptr[lvl_idx]; --row_idx) {
            VT sum = (VT)0.0;

            for (IT j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1]; ++j) {
                IT col = A_col[j];

                IF_DEBUG(
                    if (col < 0 || col >= A_n_cols)
                        SpTRSVErrorHandler::col_oob<IT>(col, j, A_n_cols););

                sum += A_val[j] * x[col];

                IF_DEBUG(if (col < row_idx) SpTRSVErrorHandler::sub_diag(
                             row_idx, col, A_val[j]););
            }

            IF_DEBUG(if (std::abs(D_val[row_idx]) < 1e-16) {
                SpTRSVErrorHandler::zero_diag();
            });

            x[row_idx] = (y[row_idx] - sum) / D_val[row_idx];
        }
    }
}

} // namespace SPTRSV_CPU
} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_LVL_CPU_IMPL_HPP