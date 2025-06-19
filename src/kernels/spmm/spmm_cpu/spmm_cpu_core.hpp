#pragma once

#include "../../../common.hpp"
#include "spmm_cpu_crs_impl.hpp"

namespace SMAX::KERNELS::SPMM::CPU {

template <typename IT, typename VT>
int initialize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags, ULL A_offset, ULL X_offset,
                        ULL Y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmm_initialize_cpu_core"));
    IF_SMAX_TIME(timers->get("initialize")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)X_offset;
    (void)Y_offset;

    IF_SMAX_TIME(timers->get("initialize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmm_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags, ULL A_offset, ULL X_offset, ULL Y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmm_apply_cpu_core"));
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
    ULL A_n_rows = args->A->crs->n_rows;
    ULL A_n_cols = args->A->crs->n_cols;
    IT *A_col = as<IT *>(args->A->crs->col);
    IT *A_row_ptr = as<IT *>(args->A->crs->row_ptr);
    VT *A_val = as<VT *>(args->A->crs->val);
    VT *X = as<VT *>(args->X->val);
    VT *Y = as<VT *>(args->Y->val);
    ULL block_vector_size = args->X->n_cols;

    if (flags->vec_row_major) {
        naive_crs_spmm_row_maj<IT, VT>(A_n_rows, A_n_cols, A_col, A_row_ptr,
                                       A_val, X + X_offset, Y + Y_offset,
                                       block_vector_size);
    } else {
        naive_crs_spmm_col_maj<IT, VT>(A_n_rows, A_n_cols, A_col, A_row_ptr,
                                       A_val, X + X_offset, Y + Y_offset,
                                       block_vector_size);
    }

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmm_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                      Flags *flags, ULL A_offset, ULL X_offset, ULL Y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmm_finalize_cpu_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)X_offset;
    (void)Y_offset;

    IF_SMAX_TIME(timers->get("finalize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmm_finalize_cpu_core"));
    return 0;
}

} // namespace SMAX::KERNELS::SPMM::CPU
