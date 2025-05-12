#ifndef SMAX_SPTRSV_CPU_CRS_HELPERS_HPP
#define SMAX_SPTRSV_CPU_CRS_HELPERS_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSV {
namespace SPTRSV_CPU {

template <typename IT, typename VT>
inline void peel_diag_crs(int A_n_rows, int A_n_cols, int A_nnz,
                          IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                          VT *RESTRICT A_val, VT *RESTRICT D_val) {

    // #pragma omp parallel for
    for (int row_idx = 0; row_idx < A_n_rows; ++row_idx) {
        for (int j = A_row_ptr[row_idx]; j < A_row_ptr[row_idx + 1]; ++j) {
            if (A_col[j] == row_idx) {
                D_val[row_idx] = A_val[j];
            }
        }
    }
};

} // namespace SPTRSV_CPU
} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_CPU_CRS_HELPERS_HPP
