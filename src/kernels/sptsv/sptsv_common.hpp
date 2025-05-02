#ifndef SPTSV_COMMON_HPP
#define SPTSV_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTSV {

class SPTSVKernelErrorHandler : public KernelErrorHandler {
  public:
    static void zero_diag() {
        const std::string message = "Zero detected on diagonal.";
        kernel_fatal("[SPTSV] " + message);
    }

    static void super_diag() {
        const std::string message = "Nonzero above diagonal detected.";
        kernel_fatal("[SPTSV] " + message);
    }

    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SPTSV");
    }
};

} // namespace SPTSV
} // namespace KERNELS
} // namespace SMAX

#endif // SPTSV_COMMON_HPP
