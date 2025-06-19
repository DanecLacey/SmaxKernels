#pragma once

#include "../spgemm_common.hpp"
#include "symbolic_phase_impl.hpp"

namespace SMAX::KERNELS::SPGEMM::CPU {

template <typename IT, typename VT>
int symbolic_phase_cpu(Timers *timers, KernelContext *k_ctx, Args *args,
                       Flags *flags) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering symbolic_phase_cpu"));
    IF_SMAX_TIME(timers->get("symbolic_phase")->start());

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

    ULL B_n_rows = args->B->crs->n_rows;
    ULL B_n_cols = args->B->crs->n_cols;
    IT *B_col = as<IT *>(args->B->crs->col);
    IT *B_row_ptr = as<IT *>(args->B->crs->row_ptr);

    // Since we want to reallocate the data pointed to by _C,
    // we need references to each of the pointers
    ULL &C_n_rows = *static_cast<ULL *>(args->C->crs->n_rows);
    ULL &C_n_cols = *static_cast<ULL *>(args->C->crs->n_cols);
    ULL &C_nnz = *static_cast<ULL *>(args->C->crs->nnz);
    IT *&C_col = as_ptr_ref<IT>(args->C->crs->col);
    IT *&C_row_ptr = as_ptr_ref<IT>(args->C->crs->row_ptr);
    VT *&C_val = as_ptr_ref<VT>(args->C->crs->val);

#if 1
    padded_symbolic_phase(timers, A_n_rows, A_col, A_row_ptr, B_n_rows,
                          B_n_cols, B_col, B_row_ptr, C_n_rows, C_n_cols, C_nnz,
                          C_col, C_row_ptr, C_val);
#endif

    IF_SMAX_TIME(timers->get("symbolic_phase")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting symbolic_phase_cpu"));
    return 0;
}

} // namespace SMAX::KERNELS::SPGEMM::CPU
