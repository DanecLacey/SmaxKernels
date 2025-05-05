#ifndef SMAX_SPMM_COMMON_HPP
#define SMAX_SPMM_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMM {

class SPMMKernelErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SPMM");
    }
};

} // namespace SPMM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMM_COMMON_HPP
