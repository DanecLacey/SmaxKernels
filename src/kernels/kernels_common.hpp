#pragma once

#include "../common.hpp"
#include <cstdarg>
#include <sstream>

namespace SMAX::KERNELS {

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

    template <typename IT, typename VT>
    static void super_diag(int row_idx, IT col, VT val,
                           const std::string &kernel_name) {
        std::ostringstream oss;
        oss << "Nonzero:" << val << " detected above diagonal at (" << row_idx
            << ", " << col << ").";
        kernel_fatal("[" + kernel_name + "] " + oss.str());
    }

    template <typename IT, typename VT>
    static void sub_diag(int row_idx, IT col, VT val,
                         const std::string &kernel_name) {
        std::ostringstream oss;
        oss << "Nonzero:" << val << " detected below diagonal at (" << row_idx
            << ", " << col << ").";
        kernel_fatal("[" + kernel_name + "] " + oss.str());
    }

    static void not_implemented(const std::string &kernel_name) {
        std::ostringstream oss;
        oss << "This kernel is not yet implemented.";
        kernel_fatal("[" + kernel_name + "] " + oss.str());
    }

    static void issue(const std::string &issue_message,
                      const std::string &kernel_name) {
        std::ostringstream oss;
        oss << "This kernel contains known issues.\n";
        oss << issue_message << "\n";
        oss << "Results not dependable.";
        kernel_warning("[" + kernel_name + "] " + oss.str());
    }
};

} // namespace SMAX::KERNELS
