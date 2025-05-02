#ifndef KERNELS_COMMON_HPP
#define KERNELS_COMMON_HPP

#include "../common.hpp"

namespace SMAX {
namespace KERNELS {

#define RESTRICT __restrict__

class KernelErrorHandler : public ErrorHandler {
  public:
    static void kernel_fatal(const std::string &message) {
        fatal("[KernelError] " + message);
    }

    static void kernel_warning(const std::string &message) {
        warning("[KernelWarning] " + message);
    }
};

} // namespace KERNELS
} // namespace SMAX

#endif // KERNELS_COMMON_HPP
