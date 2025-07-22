#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsm_common.hpp"

// Implementation files
#include "sptrsm_cpu_crs_helpers_impl.hpp"
#include "sptrsm_cpu_crs_impl.hpp"
#include "sptrsm_lvl_cpu_crs_impl.hpp" // TODO

namespace SMAX::KERNELS::SPTRSM::CPU {

template <typename IT, typename VT>
int initialize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering sptrsm_initialize_cpu_core"));
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
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting sptrsm_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering sptrsm_apply_cpu_core"));
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
    VT *X = as<VT *>(args->X->val);
    VT *Y = as<VT *>(args->Y->val);
    VT *D_val = as<VT *>(args->D->val);
    ULL block_vector_size = args->X->n_cols;

    // TODO: Don't need to split upper and lower triang into different kernels

    if (flags->vec_row_major) {
        // Lower triangular matrix is the default case
        if (flags->mat_upper_triang) {
            naive_crs_row_maj_sptrsm<false, IT, VT>(A_n_rows, A_n_cols, A_col,
                                                    A_row_ptr, A_val, D_val, X,
                                                    Y, block_vector_size);
        } else {
            // Unpermuted matrix (e.g. no lvl-set sched) is the default case
            naive_crs_row_maj_sptrsm<true, IT, VT>(A_n_rows, A_n_cols, A_col,
                                                   A_row_ptr, A_val, D_val, X,
                                                   Y, block_vector_size);
        }
    } else {
        if (flags->mat_upper_triang) {
            naive_crs_col_maj_sptrsm<false, IT, VT>(A_n_rows, A_n_cols, A_col,
                                                    A_row_ptr, A_val, D_val, X,
                                                    Y, block_vector_size);
        } else {
            naive_crs_col_maj_sptrsm<true, IT, VT>(A_n_rows, A_n_cols, A_col,
                                                   A_row_ptr, A_val, D_val, X,
                                                   Y, block_vector_size);
        }
    }

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting sptrsm_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                      Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering sptrsm_finalize_cpu_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;

    IF_SMAX_TIME(timers->get("finalize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting sptrsm_finalize_cpu_core"));
    return 0;
}

} // namespace SMAX::KERNELS::SPTRSM::CPU
