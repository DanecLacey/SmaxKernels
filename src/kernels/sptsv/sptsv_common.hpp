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
        kernel_fatal("[SPGEMMError] " + message);
    }

    static void super_diag() {
        const std::string message = "Nonzero above diagonal detected.";
        kernel_fatal("[SPGEMMError] " + message);
    }
};

} // namespace SPTSV
} // namespace KERNELS
} // namespace SMAX

#endif // SPTSV_COMMON_HPP
