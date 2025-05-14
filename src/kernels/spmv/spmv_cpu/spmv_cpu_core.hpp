#pragma once

#include "../../../common.hpp"
#include "spmv_cpu_crs_impl.hpp"

namespace SMAX::KERNELS::SPMV::SPMV_CPU {

template <typename IT, typename VT>
int initialize_cpu_core(KernelContext *k_ctx, Args *args, Flags *flags,
                        int A_offset, int x_offset, int y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmv_initialize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spmv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(KernelContext *k_ctx, Args *args, Flags *flags, int A_offset,
                   int x_offset, int y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmv_apply_cpu_core"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = args->A->n_rows;
    int A_n_cols = args->A->n_cols;
    int A_nnz = args->A->nnz;
    IT *A_col = as<IT *>(args->A->col);
    IT *A_row_ptr = as<IT *>(args->A->row_ptr);
    VT *A_val = as<VT *>(args->A->val);
    VT *x = as<VT *>(args->x->val);
    VT *y = as<VT *>(args->y->val);

#if 1
    naive_crs_spmv<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val,
                           x + x_offset, y + y_offset);
#endif

    IF_DEBUG(ErrorHandler::log("Exiting spmv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cpu_core(KernelContext *k_ctx, Args *args, Flags *flags,
                      int A_offset, int x_offset, int y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmv_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spmv_finalize_cpu_core"));
    return 0;
}

} // namespace SMAX::KERNELS::SPMV::SPMV_CPU
