#pragma once

#include "../../../common.hpp"
#include "numerical_phase_core.hpp"
#include "symbolic_phase_core.hpp"

namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU {

template <typename IT, typename VT>
int initialize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_initialize_cpu_core"));
    IF_TIME(timers->get("initialize")->start());

    // TODO
    IF_TIME(timers->get("initialize")->stop());

    IF_DEBUG(ErrorHandler::log("Exiting spgemm_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_apply_cpu_core"));
    IF_TIME(timers->get("apply")->start());

    // DL 02.05.25 NOTE: Enforcing two-phase approach Gustavson's algorithm
    symbolic_phase_cpu<IT, VT>(timers, k_ctx, args, flags);
    numerical_phase_cpu<IT, VT>(timers, k_ctx, args, flags);

    IF_TIME(timers->get("apply")->stop());
    IF_DEBUG(ErrorHandler::log("Exiting spgemm_apply_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int finalize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                      Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_finalize_cpu_core"));
    IF_TIME(timers->get("finalize")->start());

    // TODO

    IF_TIME(timers->get("finalize")->stop());
    IF_DEBUG(ErrorHandler::log("Exiting spgemm_finalize_cpu_core"));
    return 0;
};

} // namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU
