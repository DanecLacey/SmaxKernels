#ifndef SPMV_COMMON_HPP
#define SPMV_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMV {

class SPMVKernelErrorHandler : public KernelErrorHandler {
  public:
    // TODO: Kernel specific sanity checks and errors
};

} // namespace SPMV
} // namespace KERNELS
} // namespace SMAX

#endif // SPMV_COMMON_HPP
