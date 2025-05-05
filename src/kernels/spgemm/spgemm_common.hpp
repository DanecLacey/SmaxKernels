#ifndef SMAX_SPGEMM_COMMON_HPP
#define SMAX_SPGEMM_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {

class SPGEMMKernelErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SPGEMM");
    }

    static void multithreaded_issue() {
        KernelErrorHandler::issue("Multithreaded problems", "SPGEMM");
    }
};

} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMM_COMMON_HPP
