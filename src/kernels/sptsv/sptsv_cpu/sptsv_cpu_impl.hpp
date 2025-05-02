#ifndef SPTSV_CPU_IMPL_HPP
#define SPTSV_CPU_IMPL_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptsv_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTSV {
namespace SPTSV_CPU {

// TODO: JH
// // Saad: Iterative Methods for Sparse Linear Systems (ch 11.6)
// void spltsv_lvl(

// ){
//     // TODO
// }

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
inline void basic_sptsv(IT A_n_rows, IT A_n_cols, IT A_nnz, IT *RESTRICT A_col,
                        IT *RESTRICT A_row_ptr, VT *RESTRICT A_val,
                        VT *RESTRICT X, VT *RESTRICT Y) {
    for (IT i = 0; i < A_n_rows; ++i) {
        VT sum = 0.0;
        VT diag = 0.0;

        for (IT idx = A_row_ptr[i]; idx < A_row_ptr[i + 1]; ++idx) {
            IT j = A_col[idx];
            VT val = A_val[idx];

            if (j < i) {
                sum += val * X[j];
            } else if (j == i) {
                diag = val;
            } else {
                IF_DEBUG(SPTSVKernelErrorHandler::super_diag());
            }
        }

        IF_DEBUG(
            if (abs(diag) < 1e-16) { SPTSVKernelErrorHandler::zero_diag(); });

        X[i] = (Y[i] - sum) / diag;
    }
}

} // namespace SPTSV_CPU
} // namespace SPTSV
} // namespace KERNELS
} // namespace SMAX

#endif // SPTSV_CPU_IMPL_HPP