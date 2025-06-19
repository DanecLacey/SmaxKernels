#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX::KERNELS::SPTRSV::CPU {

template <typename IT, typename VT>
inline void peel_diag_crs(ULL n_rows, IT *col, IT *row_ptr, VT *val, VT *D) {

    for (ULL row_idx = 0; row_idx < n_rows; ++row_idx) {
        IT row_start = row_ptr[row_idx];
        IT row_end = row_ptr[row_idx + 1] - 1;
        long long int diag_j = -1; // Init diag col

        // find the diag in this row_idx (since row need not be col sorted)
        for (IT j = row_start; j <= row_end; ++j) {
            if (col[j] == (IT)row_idx) {
                diag_j = j;
                D[row_idx] = val[j]; // extract
                if (std::abs(D[row_idx]) < 1e-16) {
                    SpTRSVErrorHandler::zero_diag(row_idx);
                }
            }
        }
        if (diag_j < 0) {
            SpTRSVErrorHandler::no_diag(row_idx);
        }

        // if it's not already at the end, swap it into the last slot
        if (diag_j != row_end) {
            std::swap(col[diag_j], col[row_end]);
            std::swap(val[diag_j], val[row_end]);
        }
    };
}

} // namespace SMAX::KERNELS::SPTRSV::CPU
