#ifndef SMAX_SPGEMV_CPU_CRS_IMPL_HPP
#define SMAX_SPGEMV_CPU_CRS_IMPL_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spgemv_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMV {
namespace SPGEMV_CPU {

template <typename IT, typename VT>
inline void
naive_crs_coo_spgemv(int A_n_rows, int A_n_cols, int A_nnz, IT *RESTRICT A_col,
                     IT *RESTRICT A_row_ptr, VT *RESTRICT A_val, int x_n_rows,
                     int x_nnz, IT *RESTRICT x_idx, VT *RESTRICT x_val,
                     int &y_n_rows, int &y_nnz, IT *&y_idx, VT *&y_val) {

    SpGEMVErrorHandler::not_implemented();
}

} // namespace SPGEMV_CPU
} // namespace SPGEMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMV_CPU_CRS_IMPL_HPP