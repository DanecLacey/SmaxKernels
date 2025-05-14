#pragma once

#include "../spgemm_common.hpp"
#include "numerical_phase_impl.hpp"

namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU {

template <typename IT, typename VT>
int numerical_phase_cpu(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering numerical_phase_cpu"));
    IF_TIME(timers->get("numerical_phase")->start());

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = args->A->n_rows;
    int A_n_cols = args->A->n_cols;
    int A_nnz = args->A->nnz;
    IT *A_col = as<IT *>(args->A->col);
    IT *A_row_ptr = as<IT *>(args->A->row_ptr);
    VT *A_val = as<VT *>(args->A->val);

    int B_n_rows = args->B->n_rows;
    int B_n_cols = args->B->n_cols;
    int B_nnz = args->B->nnz;
    IT *B_col = as<IT *>(args->B->col);
    IT *B_row_ptr = as<IT *>(args->B->row_ptr);
    VT *B_val = as<VT *>(args->B->val);

    int C_n_rows = *args->C->n_rows;
    int C_n_cols = *args->C->n_cols;
    int C_nnz = *args->C->nnz;
    IT *C_col = as<IT *>(args->C->col);
    IT *C_row_ptr = as<IT *>(args->C->row_ptr);
    VT *C_val = as<VT *>(args->C->val);

#if 1
    basic_numerical_phase(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val,
                          B_n_rows, B_n_cols, B_nnz, B_col, B_row_ptr, B_val,
                          C_n_rows, C_n_cols, C_nnz, C_col, C_row_ptr, C_val);
#endif

    IF_TIME(timers->get("numerical_phase")->stop());
    IF_DEBUG(ErrorHandler::log("Exiting numerical_phase_cpu"));
    return 0;
}

} // namespace SMAX::KERNELS::SPGEMM::SPGEMM_CPU
