#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

// Implementation files
#include "sptrsv_cpu_crs_helpers_impl.hpp"
#include "sptrsv_cpu_crs_impl.hpp"
#include "sptrsv_lvl_cpu_crs_impl.hpp"

namespace SMAX::KERNELS::SPTRSV::CPU {

template <typename IT, typename VT>
int initialize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering sptrsv_initialize_cpu_core"));
    IF_SMAX_TIME(timers->get("initialize")->start());

    if (!flags->diag_collected) {
        // Cast void pointers to the correct types with "as"
        // Dereference to get usable data
        ULL A_n_rows = args->A->crs->n_rows;
        IT *A_col = as<IT *>(args->A->crs->col);
        IT *A_row_ptr = as<IT *>(args->A->crs->row_ptr);
        VT *A_val = as<VT *>(args->A->crs->val);

        // Resize D to match A_n_rows x 1
        args->D->allocate_internal(A_n_rows, 1, sizeof(VT));

        // Get typed pointer to internal data
        VT *D_val = as<VT *>(args->D->val);

        peel_diag_crs<IT, VT>(A_n_rows, A_col, A_row_ptr, A_val, D_val);

        flags->diag_collected = true;
    }

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;

    IF_SMAX_TIME(timers->get("initialize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting sptrsv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering sptrsv_apply_cpu_core"));
    IF_SMAX_TIME(timers->get("apply")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    ULL A_n_rows = args->A->crs->n_rows;
    ULL A_n_cols = args->A->crs->n_cols;
    IT *A_col = as<IT *>(args->A->crs->col);
    IT *A_row_ptr = as<IT *>(args->A->crs->row_ptr);
    VT *A_val = as<VT *>(args->A->crs->val);
    VT *x = as<VT *>(args->x->val);
    VT *y = as<VT *>(args->y->val);
    VT *D_val = as<VT *>(args->D->val);

    if (flags->mat_permuted) {
        int *lvl_ptr = args->uc->lvl_ptr;
        int n_levels = args->uc->n_levels;
        if (flags->mat_upper_triang) {
            crs_sptrsv_lvl<false, IT, VT>(n_levels, A_n_cols, A_col, A_row_ptr,
                                          A_val, D_val, x, y, lvl_ptr);
        } else {
            crs_sptrsv_lvl<true, IT, VT>(n_levels, A_n_cols, A_col, A_row_ptr,
                                         A_val, D_val, x, y, lvl_ptr);
        }

    } else {
        // Lower triangular matrix is the default case
        if (flags->mat_upper_triang) {
            naive_crs_sptrsv<false, IT, VT>(A_n_rows, A_n_cols, A_col,
                                            A_row_ptr, A_val, D_val, x, y);
        } else {
            // Unpermuted matrix (e.g. no lvl-set sched) is the default case
            naive_crs_sptrsv<true, IT, VT>(A_n_rows, A_n_cols, A_col, A_row_ptr,
                                           A_val, D_val, x, y);
        }
    }

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting sptrsv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                      Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering sptrsv_finalize_cpu_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;

    IF_SMAX_TIME(timers->get("finalize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting sptrsv_finalize_cpu_core"));
    return 0;
}

} // namespace SMAX::KERNELS::SPTRSV::CPU
