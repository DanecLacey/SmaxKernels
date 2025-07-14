#pragma once

#include "../../../common.hpp"
#include "bspmv_cpu_bcrs_impl.hpp"

namespace SMAX::KERNELS::BSPMV::CPU {

template <typename IT, typename VT>
int initialize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags, ULL A_offset, ULL x_offset,
                        ULL y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering bspmv_initialize_cpu_core"));
    IF_SMAX_TIME(timers->get("initialize")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    IF_SMAX_TIME(timers->get("initialize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting bspmv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags, ULL A_offset, ULL x_offset, ULL y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering bspmv_apply_cpu_core"));
    IF_SMAX_TIME(timers->get("apply")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
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
    if (flags->is_block_column_major) {
        naive_bcrs_spmv<IT, VT, true>(
            args->A->crs->n_rows,
            args->A->crs->n_cols,
            args->A->crs->b_height,
            args->A->crs->b_width,
            args->A->crs->height_pad,
            args->A->crs->width_pad,
            as<IT *>(args->A->crs->col),
            as<IT *>(args->A->crs->row_ptr),
            as<VT *>(args->A->crs->val),
            x + x_offset,
            y + y_offset);
    } else {
        naive_bcrs_spmv<IT, VT, false>(
            args->A->crs->n_rows,
            args->A->crs->n_cols,
            args->A->crs->b_height,
            args->A->crs->b_width,
            args->A->crs->height_pad,
            args->A->crs->width_pad,
            as<IT *>(args->A->crs->col),
            as<IT *>(args->A->crs->row_ptr),
            as<VT *>(args->A->crs->val),
            x + x_offset,
            y + y_offset);
    }
    // clang-format on

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting bspmv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                      Flags *flags, ULL A_offset, ULL x_offset, ULL y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering bspmv_finalize_cpu_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
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

} // namespace SMAX::KERNELS::SPMV::CPU