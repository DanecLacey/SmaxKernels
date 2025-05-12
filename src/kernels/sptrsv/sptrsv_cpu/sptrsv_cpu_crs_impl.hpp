#ifndef SMAX_SPTRSV_CPU_IMPL_HPP
#define SMAX_SPTRSV_CPU_IMPL_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSV {
namespace SPTRSV_CPU {

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
                              VT *RESTRICT A_val, VT *RESTRICT X,
                              VT *RESTRICT Y) {
    for (int i = 0; i < A_n_rows; ++i) {
        VT sum = 0.0;
        VT diag = 0.0;

        for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
            IF_DEBUG(
                if (A_col[j] < 0 || A_col[j] >= A_n_cols)
                    SpTRSVErrorHandler::col_oob<IT>(A_col[j], j, A_n_cols););
            VT val = A_val[j];

            if (A_col[j] < i) {
                sum += val * X[A_col[j]];
            } else if (A_col[j] == i) {
                diag = val;
            } else {
                IF_DEBUG(
                    printf("row: %d, col: %d, val: %f\n", i, A_col[j], val));
                IF_DEBUG(SpTRSVErrorHandler::super_diag());
            }
        }

        IF_DEBUG(if (abs(diag) < 1e-16) { SpTRSVErrorHandler::zero_diag(); });

        X[i] = (Y[i] - sum) / diag;
    }
}

template <typename IT, typename VT>
inline void naive_crs_sputrsv(int A_n_rows, int A_n_cols, int A_nnz,
                              IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                              VT *RESTRICT A_val, VT *RESTRICT X,
                              VT *RESTRICT Y) {
    for (int i = A_n_rows - 1; i >= 0; --i) {
        VT sum = 0.0;
        VT diag = 0.0;

        for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
            IF_DEBUG(
                if (A_col[j] < 0 || A_col[j] >= A_n_cols)
                    SpTRSVErrorHandler::col_oob<IT>(A_col[j], j, A_n_cols););
            VT val = A_val[j];

            if (A_col[j] < i) {
                sum += val * X[A_col[j]];
            } else if (A_col[j] == i) {
                diag = val;
            } else {
                IF_DEBUG(
                    printf("row: %d, col: %d, val: %f\n", i, A_col[j], val));
                IF_DEBUG(SpTRSVErrorHandler::super_diag());
            }
        }

        IF_DEBUG(if (abs(diag) < 1e-16) { SpTRSVErrorHandler::zero_diag(); });

        X[i] = (Y[i] - sum) / diag;
    }
}

} // namespace SPTRSV_CPU
} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_CPU_IMPL_HPP