#pragma once

#include "../spgemm_common.hpp"
#include "lvl_fused_impl.hpp"

namespace SMAX::KERNELS::SPGEMM::CPU {

template <typename IT, typename VT>
int lvl_fused_cpu(Timers *timers, KernelContext *k_ctx, Args *args,
                  Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering lvl_fused_cpu"));
    IF_SMAX_TIME(timers->get("Lvl")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)flags;

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    ULL A_n_rows = args->A->crs->n_rows;
    ULL A_n_cols = args->A->crs->n_cols;
    ULL A_nnz = args->A->crs->nnz;
    IT *A_col = as<IT *>(args->A->crs->col);
    IT *A_row_ptr = as<IT *>(args->A->crs->row_ptr);
    VT *A_val = as<VT *>(args->A->crs->val);

    ULL B_n_rows = args->B->crs->n_rows;
    ULL B_n_cols = args->B->crs->n_cols;
    ULL B_nnz = args->B->crs->nnz;
    IT *B_col = as<IT *>(args->B->crs->col);
    IT *B_row_ptr = as<IT *>(args->B->crs->row_ptr);
    VT *B_val = as<VT *>(args->B->crs->val);

    // Since we want to reallocate the data pointed to by _C,
    // we need references to each of the pointers
    ULL &C_n_rows = *static_cast<ULL *>(args->C->crs->n_rows);
    ULL &C_n_cols = *static_cast<ULL *>(args->C->crs->n_cols);
    ULL &C_nnz = *static_cast<ULL *>(args->C->crs->nnz);
    IT *&C_col = as_ptr_ref<IT>(args->C->crs->col);
    IT *&C_row_ptr = as_ptr_ref<IT>(args->C->crs->row_ptr);
    VT *&C_val = as_ptr_ref<VT>(args->C->crs->val);

    // Assumed to be collected earlier
    int *lvl_ptr = args->uc->lvl_ptr;
    int n_levels = args->uc->n_levels;

#if 1
    lvl_seq_traversal(timers, A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                      A_val, B_n_rows, B_n_cols, B_nnz, B_col, B_row_ptr, B_val,
                      C_n_rows, C_n_cols, C_nnz, C_col, C_row_ptr, C_val,
                      lvl_ptr, n_levels);
#endif

    IF_SMAX_TIME(timers->get("Lvl")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting lvl_fused_cpu"));
    return 0;
}

} // namespace SMAX::KERNELS::SPGEMM::CPU