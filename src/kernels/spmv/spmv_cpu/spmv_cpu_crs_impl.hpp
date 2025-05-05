#ifndef SMAX_SPMV_CPU_IMPL_HPP
#define SMAX_SPMV_CPU_IMPL_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmv_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMV {
namespace SPMV_CPU {

template <typename IT, typename VT>
inline void naive_crs_spmv(int A_n_rows, int A_n_cols, int A_nnz,
                           IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                           VT *RESTRICT A_val, VT *RESTRICT X, VT *RESTRICT Y) {

#pragma omp parallel for schedule(static)
    for (int row = 0; row < A_n_rows; ++row) {
        VT sum{};

#pragma omp simd
        for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j) {

            IF_DEBUG(
#if DEBUG_LEVEL == 3
                printf("A_val[%d] = %f\n", j, A_val[j]);
                printf("A_col[%d] = %d\n", j, A_col[j]);
                printf("X[A_col[%d]] = %f\n", j, X[A_col[j]]);
#endif
                if (A_col[j] < 0 || A_col[j] >= (IT)A_n_cols)
                    SPMVKernelErrorHandler::col_oob<IT>(A_col[j], j,
                                                        A_n_cols););

            sum += A_val[j] * X[A_col[j]];
        }
        Y[row] = sum;
    }
}

} // namespace SPMV_CPU
} // namespace SPMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMV_CPU_IMPL_HPP