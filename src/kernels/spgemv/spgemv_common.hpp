#ifndef SMAX_SPGEMV_COMMON_HPP
#define SMAX_SPGEMV_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMV {

class SPGEMVKernelErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SPGEMV");
    }

    static void not_implemented() {
        KernelErrorHandler::not_implemented("SPGEMV");
    }
};

} // namespace SPGEMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMV_COMMON_HPP
