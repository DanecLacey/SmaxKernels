#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsm_common.hpp"

namespace SMAX::KERNELS::SPTRSM::SPTRSM_CPU {

template <typename IT, typename VT>
inline void peel_diag_crs(int A_n_rows, int A_n_cols, int A_nnz, IT *A_col,
                          IT *A_row_ptr, VT *A_val, VT *D_val) {

    for (int row_idx = 0; row_idx < A_n_rows; ++row_idx) {
        int row_start = A_row_ptr[row_idx];
        int row_end = A_row_ptr[row_idx + 1] - 1;
        int diag_j = -1; // Init diag col

        // find the diag in this row_idx (since row need not be col sorted)
        for (int j = row_start; j <= row_end; ++j) {
            if (A_col[j] == row_idx) {
                diag_j = j;
                D_val[row_idx] = A_val[j]; // extract
                if (std::abs(D_val[row_idx]) < 1e-16) {
                    SpTRSMErrorHandler::zero_diag(row_idx);
                }
            }
        }
        if (diag_j < 0) {
            SpTRSMErrorHandler::no_diag(row_idx);
        }

        // if it's not already at the end, swap it into the last slot
        if (diag_j != row_end) {
            std::swap(A_col[diag_j], A_col[row_end]);
            std::swap(A_val[diag_j], A_val[row_end]);
        }
    };
}

} // namespace SMAX::KERNELS::SPTRSM::SPTRSM_CPU
