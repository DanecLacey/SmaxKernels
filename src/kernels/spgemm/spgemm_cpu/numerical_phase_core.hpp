#pragma once

#include "../spgemm_common.hpp"
#include "numerical_phase_impl.hpp"

namespace SMAX::KERNELS::SPGEMM::CPU {

template <typename IT, typename VT>
int numerical_phase_cpu(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering numerical_phase_cpu"));
    IF_SMAX_TIME(timers->get("numerical_phase")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)flags;

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    ULL A_n_rows = args->A->crs->n_rows;
    IT *A_col = as<IT *>(args->A->crs->col);
    IT *A_row_ptr = as<IT *>(args->A->crs->row_ptr);
    VT *A_val = as<VT *>(args->A->crs->val);

    IT *B_col = as<IT *>(args->B->crs->col);
    IT *B_row_ptr = as<IT *>(args->B->crs->row_ptr);
    VT *B_val = as<VT *>(args->B->crs->val);

    ULL C_n_cols = *static_cast<ULL *>(args->C->crs->n_cols);
    IT *&C_col = as_ptr_ref<IT>(args->C->crs->col);
    IT *&C_row_ptr = as_ptr_ref<IT>(args->C->crs->row_ptr);
    VT *&C_val = as_ptr_ref<VT>(args->C->crs->val);

#if 1
    basic_numerical_phase(timers, A_n_rows, A_col, A_row_ptr, A_val, B_col,
                          B_row_ptr, B_val, C_n_cols, C_col, C_row_ptr, C_val);
#endif

    IF_SMAX_TIME(timers->get("numerical_phase")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting numerical_phase_cpu"));
    return 0;
}

} // namespace SMAX::KERNELS::SPGEMM::CPU
