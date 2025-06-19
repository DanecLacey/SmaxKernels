#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmv_common.hpp"

namespace SMAX::KERNELS::SPMV::CPU {

template <typename IT, typename VT>
inline void naive_scs_spmv(const ULL C, const ULL n_cols, const ULL n_chunks,
                           const IT *RESTRICT chunk_ptr,
                           const IT *RESTRICT A_chunk_lengths,
                           const IT *RESTRICT A_col, const VT *RESTRICT val,
                           const VT *RESTRICT x, VT *RESTRICT y) {

    // clang-format off
#pragma omp parallel for schedule(static)
    for (ULL c = 0; c < n_chunks; ++c) {

        VT tmp[C];
        for (ULL i = 0; i < C; ++i) {
            tmp[i] = VT{};
        }

        IT cs = chunk_ptr[c];

        for (IT j = 0; j < A_chunk_lengths[c]; ++j) {
            IT offset = cs + j * C;
            for (ULL i = 0; i < C; ++i) {
                IT col = A_col[offset + i];

                IF_SMAX_DEBUG(
                    if (col < 0 || col >= (IT)n_cols)
                        SpMVErrorHandler::col_oob<IT>(col, offset + i, n_cols););
                IF_SMAX_DEBUG_3(SpMVErrorHandler::print_crs_elem<IT, VT>(
                                    val[j], col, x[col], offset + i););

                tmp[i] += val[offset + i] * x[col];
            }
        }

        for (ULL i = 0; i < C; ++i) {
            y[c * C + i] = tmp[i];
        }
    }

    // clang-format on
}

} // namespace SMAX::KERNELS::SPMV::CPU
