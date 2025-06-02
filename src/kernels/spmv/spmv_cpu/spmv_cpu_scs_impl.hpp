#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmv_common.hpp"

namespace SMAX::KERNELS::SPMV::SPMV_CPU {

template <typename IT, typename VT>
inline void naive_scs_spmv(const int C, const int A_n_cols,
                           const int A_n_chunks, const IT *RESTRICT A_chunk_ptr,
                           const IT *RESTRICT A_chunk_lengths,
                           const IT *RESTRICT A_col, const VT *RESTRICT A_val,
                           const VT *RESTRICT x, VT *RESTRICT y) {

    // clang-format off
#pragma omp parallel for schedule(static)
    for (int c = 0; c < A_n_chunks; ++c) {

        VT tmp[C];
        for (int i = 0; i < C; ++i) {
            tmp[i] = VT{};
        }

        IT cs = A_chunk_ptr[c];

        for (IT j = 0; j < A_chunk_lengths[c]; ++j) {
            IT offset = cs + j * C;
            for (int i = 0; i < C; ++i) {
                IT col = A_col[offset + i];

                IF_SMAX_DEBUG(
                    if (col < 0 || col >= (IT)A_n_cols)
                        SpMVErrorHandler::col_oob<IT>(col, offset + i, A_n_cols););
                IF_SMAX_DEBUG_3(SpMVErrorHandler::print_crs_elem<IT, VT>(
                                    A_val[j], col, x[col], offset + i););

                tmp[i] += A_val[offset + i] * x[col];
            }
        }

        for (int i = 0; i < C; ++i) {
            y[c * C + i] = tmp[i];
        }
    }

    // clang-format on
}

} // namespace SMAX::KERNELS::SPMV::SPMV_CPU
