#pragma once

#include "../spgemm_common.hpp"
#include "numerical_phase_impl.hpp"

namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU {

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
    int A_n_rows = args->A->n_rows;
    IT *A_col = as<IT *>(args->A->col);
    IT *A_row_ptr = as<IT *>(args->A->row_ptr);
    VT *A_val = as<VT *>(args->A->val);

    IT *B_col = as<IT *>(args->B->col);
    IT *B_row_ptr = as<IT *>(args->B->row_ptr);
    VT *B_val = as<VT *>(args->B->val);

    int C_n_cols = *static_cast<int *>(args->C->n_cols);
    IT *&C_col = as_ptr_ref<IT>(args->C->col);
    IT *&C_row_ptr = as_ptr_ref<IT>(args->C->row_ptr);
    VT *&C_val = as_ptr_ref<VT>(args->C->val);

#if 1
    basic_numerical_phase(A_n_rows, A_col, A_row_ptr, A_val, B_col, B_row_ptr,
                          B_val, C_n_cols, C_col, C_row_ptr, C_val);
#endif

    IF_SMAX_TIME(timers->get("numerical_phase")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting numerical_phase_cpu"));
    return 0;
}

} // namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU
