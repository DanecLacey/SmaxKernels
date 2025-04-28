#ifndef SPGEMM_COMMON_HPP
#define SPGEMM_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {

class SPGEMMKernelErrorHandler : public KernelErrorHandler {
  public:
    // TODO: Kernel specific sanity checks and errors
};

} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // SPGEMM_COMMON_HPP
