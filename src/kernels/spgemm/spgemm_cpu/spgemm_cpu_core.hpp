#pragma once

#include "../../../common.hpp"
#include "numerical_phase_core.hpp"
#include "symbolic_phase_core.hpp"

namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU {

template <typename IT, typename VT>
int initialize_cpu_core(KernelContext *k_ctx, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_initialize_cpu_core"));
    // TODO

    IF_DEBUG(ErrorHandler::log("Exiting spgemm_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(KernelContext *k_ctx, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_apply_cpu_core"));

    // DL 02.05.25 NOTE: Enforcing two-phase approach Gustavson's algorithm
    symbolic_phase_cpu<IT, VT>(k_ctx, args, flags);
    numerical_phase_cpu<IT, VT>(k_ctx, args, flags);

    IF_DEBUG(ErrorHandler::log("Exiting spgemm_apply_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int finalize_cpu_core(KernelContext *k_ctx, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemm_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spgemm_finalize_cpu_core"));
    return 0;
};

} // namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU
