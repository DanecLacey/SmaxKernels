#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmv_common.hpp"

namespace SMAX::KERNELS::SPMV::SPMV_CPU {

template <typename IT, typename VT>
inline void naive_crs_spmv(int A_n_rows, int A_n_cols, IT *RESTRICT A_col,
                           IT *RESTRICT A_row_ptr, VT *RESTRICT A_val,
                           VT *RESTRICT x, VT *RESTRICT y) {

    // clang-format off
#pragma omp parallel for schedule(static)
    for (int row = 0; row < A_n_rows; ++row) {
        VT sum{};

#pragma omp simd
        for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j) {
            IT col = A_col[j];

            IF_SMAX_DEBUG(
                if (col < 0 || col >= (IT)A_n_cols)
                    SpMVErrorHandler::col_oob<IT>(col, j, A_n_cols);
            );
            IF_SMAX_DEBUG_3(
                SpMVErrorHandler::print_crs_elem<IT, VT>(
                    A_val[j], col, x[col], j);
            );
            
            sum += A_val[j] * x[col];
        }
        y[row] = sum;
    }
    IF_SMAX_DEBUG_3(printf("Finish SpMV\n"));
    // clang-format on
}

} // namespace SMAX::KERNELS::SPMV::SPMV_CPU
