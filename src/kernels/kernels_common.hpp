#ifndef KERNELS_COMMON_HPP
#define KERNELS_COMMON_HPP

#include "../common.hpp"
#include <sstream>

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

    template <typename IT>
    static void col_oob(IT col_value, int j, int max_cols,
                        const std::string &kernel_name) {
        std::ostringstream oss;
        oss << "Column index " << col_value << " at position " << j
            << " is out of bounds (max = " << max_cols - 1 << ").";
        kernel_fatal("[" + kernel_name + "] " + oss.str());
    }
};

} // namespace KERNELS
} // namespace SMAX

#endif // KERNELS_COMMON_HPP
