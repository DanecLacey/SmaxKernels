#ifndef SMAX_SPGEMM_CPU_CORE_HPP
#define SMAX_SPGEMM_CPU_CORE_HPP

#include "../../../common.hpp"
#include "numerical_phase_core.hpp"
#include "symbolic_phase_core.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {
namespace SPGEMM_CPU {

template <typename IT, typename VT>
int spgemm_initialize_cpu_core(KernelContext context, Args *args,
                               Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_initialize_cpu_core"));
    // TODO

    IF_DEBUG(ErrorHandler::log("Exiting spgemm_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spgemm_apply_cpu_core(KernelContext context, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_apply_cpu_core"));

    // DL 02.05.25 NOTE: Enforcing two-phase approach Gustavson's algorithm
    symbolic_phase_cpu<IT, VT>(context, args, flags);
    numerical_phase_cpu<IT, VT>(context, args, flags);

    IF_DEBUG(ErrorHandler::log("Exiting spgemm_apply_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spgemm_finalize_cpu_core(KernelContext context, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spgemm_finalize_cpu_core"));
    return 0;
};

} // namespace SPGEMM_CPU
} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMM_CPU_CORE_HPP
