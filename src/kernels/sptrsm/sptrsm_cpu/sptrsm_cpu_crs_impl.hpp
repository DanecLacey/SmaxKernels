#ifndef SMAX_SPTRSM_CPU_IMPL_HPP
#define SMAX_SPTRSM_CPU_IMPL_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsm_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSM {
namespace SPTRSM_CPU {

template <typename IT, typename VT>
inline void native_crs_sptrsm(int A_n_rows, int A_n_cols, int A_nnz,
                              IT *RESTRICT A_col, IT *RESTRICT A_row_ptr,
                              VT *RESTRICT A_val, VT *RESTRICT X,
                              VT *RESTRICT Y, int block_vector_size) {
    SpTRSMErrorHandler::not_implemented();
}

} // namespace SPTRSM_CPU
} // namespace SPTRSM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSM_CPU_IMPL_HPP