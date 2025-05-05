#ifndef SMAX_SPMV_COMMON_HPP
#define SMAX_SPMV_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMV {

class SPMVKernelErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SPMV");
    }
};

} // namespace SPMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMV_COMMON_HPP
