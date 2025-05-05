#ifndef SMAX_SPTRSV_COMMON_HPP
#define SMAX_SPTRSV_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSV {

class SPTRSVKernelErrorHandler : public KernelErrorHandler {
  public:
    static void zero_diag() {
        const std::string message = "Zero detected on diagonal.";
        kernel_fatal("[SPTRSV] " + message);
    }

    static void super_diag() {
        const std::string message = "Nonzero above diagonal detected.";
        kernel_fatal("[SPTRSV] " + message);
    }

    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SPTRSV");
    }
};

} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_COMMON_HPP
