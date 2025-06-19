#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmv_common.hpp"

namespace SMAX::KERNELS::SPMV::CPU {

template <typename IT, typename VT>
inline void naive_crs_spmv(const ULL n_rows, const ULL n_cols,
                           const IT *RESTRICT col, const IT *RESTRICT row_ptr,
                           const VT *RESTRICT val, const VT *RESTRICT x,
                           VT *RESTRICT y) {

    // clang-format off
#pragma omp parallel for schedule(static)
    for (ULL row = 0; row < n_rows; ++row) {
        VT sum{};

#pragma omp simd
        for (IT j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {

            IF_SMAX_DEBUG(
                if (col[j] < 0 || col[j] >= (IT)n_cols)
                    SpMVErrorHandler::col_oob<IT>(col[j], j, n_cols);
            );
            IF_SMAX_DEBUG_3(
                SpMVErrorHandler::print_crs_elem<IT, VT>(
                    val[j], col, x[col[j]], j);
            );
            
            sum += val[j] * x[col[j]];
        }
        y[row] = sum;
    }
    IF_SMAX_DEBUG_3(printf("Finish SpMV\n"));
    // clang-format on
}

} // namespace SMAX::KERNELS::SPMV::CPU
