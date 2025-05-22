#pragma once

#include "../../../common.hpp"
#include "numerical_phase_core.hpp"
#include "symbolic_phase_core.hpp"

namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU {

template <typename IT, typename VT>
int initialize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spgemm_initialize_cpu_core"));
    IF_SMAX_TIME(timers->get("initialize")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;

    IF_SMAX_TIME(timers->get("initialize")->stop());

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spgemm_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spgemm_apply_cpu_core"));
    IF_SMAX_TIME(timers->get("apply")->start());

    // DL 02.05.25 NOTE: Enforcing two-phase approach Gustavson's algorithm
    symbolic_phase_cpu<IT, VT>(timers, k_ctx, args, flags);
    numerical_phase_cpu<IT, VT>(timers, k_ctx, args, flags);

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spgemm_apply_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int finalize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                      Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spgemm_finalize_cpu_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;

    IF_SMAX_TIME(timers->get("finalize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spgemm_finalize_cpu_core"));
    return 0;
};

} // namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU
