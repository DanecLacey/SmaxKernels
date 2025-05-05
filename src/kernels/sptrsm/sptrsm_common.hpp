#ifndef SMAX_SPTRSM_COMMON_HPP
#define SMAX_SPTRSM_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSM {

class SPTRSMKernelErrorHandler : public KernelErrorHandler {
  public:
    static void zero_diag() {
        const std::string message = "Zero detected on diagonal.";
        kernel_fatal("[SPTRSM] " + message);
    }

    static void super_diag() {
        const std::string message = "Nonzero above diagonal detected.";
        kernel_fatal("[SPTRSM] " + message);
    }

    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SPTRSM");
    }

    static void not_implemented() {
        KernelErrorHandler::not_implemented("SPTRSM");
    }
};

} // namespace SPTRSM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSM_COMMON_HPP
