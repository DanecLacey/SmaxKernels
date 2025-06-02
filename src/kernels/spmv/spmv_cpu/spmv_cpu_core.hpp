#pragma once

#include "../../../common.hpp"
#include "spmv_cpu_crs_impl.hpp"
#include "spmv_cpu_scs_impl.hpp"

namespace SMAX::KERNELS::SPMV::SPMV_CPU {

template <typename IT, typename VT>
int initialize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags, int A_offset, int x_offset,
                        int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_initialize_cpu_core"));
    IF_SMAX_TIME(timers->get("initialize")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    IF_SMAX_TIME(timers->get("initialize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags, int A_offset, int x_offset, int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_apply_cpu_core"));
    IF_SMAX_TIME(timers->get("apply")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)flags;
    (void)A_offset;

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    VT *x = as<VT *>(args->x->val);
    VT *y = as<VT *>(args->y->val);

    // clang-format off
    if (flags->is_mat_scs) {
        naive_scs_spmv<IT, VT>(
            args->A->scs->C,
            args->A->scs->n_cols,
            args->A->scs->n_chunks,
            as<IT *>(args->A->scs->chunk_ptr),
            as<IT *>(args->A->scs->chunk_lengths),
            as<IT *>(args->A->scs->col),
            as<VT *>(args->A->scs->val),
            x + x_offset,
            y + y_offset);
    } else {
        naive_crs_spmv<IT, VT>(
            args->A->crs->n_rows,
            args->A->crs->n_cols,
            as<IT *>(args->A->crs->col),
            as<IT *>(args->A->crs->row_ptr),
            as<VT *>(args->A->crs->val),
            x + x_offset,
            y + y_offset);
    }
    // clang-format on

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                      Flags *flags, int A_offset, int x_offset, int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_finalize_cpu_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    IF_SMAX_TIME(timers->get("finalize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_finalize_cpu_core"));
    return 0;
}

} // namespace SMAX::KERNELS::SPMV::SPMV_CPU
