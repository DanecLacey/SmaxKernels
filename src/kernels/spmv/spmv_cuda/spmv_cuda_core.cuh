#pragma once

#include "../../../common.hpp"
#include "../../../platforms/cuda/cuda.cuh"
#include "spmv_cuda_crs_impl.cuh"

namespace SMAX::KERNELS::SPMV::SPMV_CUDA {

template <typename IT, typename VT>
int initialize_cuda_core(Timers *timers, KernelContext *k_ctx, Args *args,
                         Flags *flags, int A_offset, int x_offset,
                         int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_initialize_cuda_core"));
    IF_SMAX_TIME(timers->get("initialize")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = args->A->n_rows;
    int A_nnz = args->A->nnz;
    IT *A_col = as<IT *>(args->A->col);
    IT *A_row_ptr = as<IT *>(args->A->row_ptr);
    VT *A_val = as<VT *>(args->A->val);
    VT *x = as<VT *>(args->x->val);
    VT *y = as<VT *>(args->y->val);

    // IT *d_A_col = as<IT *>(args->d_A->col);
    // IT *d_A_row_ptr = as<IT *>(args->d_A->row_ptr);
    // VT *d_A_val = as<VT *>(args->d_A->val);
    // VT *d_x = as<VT *>(args->d_x->val);
    // VT *d_y = as<VT *>(args->d_y->val);

    // Get typed pointers from device
    transfer_HtoD<IT>(A_col, args->d_A->col, A_nnz);
    transfer_HtoD<IT>(A_row_ptr, args->d_A->row_ptr, A_n_rows + 1);
    transfer_HtoD<VT>(A_val, args->d_A->val, A_nnz);
    transfer_HtoD<VT>(x, args->d_x->val, A_n_rows);
    transfer_HtoD<VT>(y, args->d_y->val, A_n_rows);

    // Copy metadata from host matrix
    args->d_A->n_rows = args->A->n_rows;
    args->d_A->n_cols = args->A->n_cols;
    args->d_A->nnz = args->A->nnz;

    IF_SMAX_TIME(timers->get("initialize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_initialize_cuda_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cuda_core(Timers *timers, KernelContext *k_ctx, Args *args,
                    Flags *flags, int A_offset, int x_offset, int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_apply_cuda_core"));
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
    int d_A_n_rows = args->d_A->n_rows;
    IT *d_A_col = as<IT *>(args->d_A->col);
    IT *d_A_row_ptr = as<IT *>(args->d_A->row_ptr);
    VT *d_A_val = as<VT *>(args->d_A->val);
    VT *d_x = as<VT *>(args->d_x->val);
    VT *d_y = as<VT *>(args->d_y->val);

#if 1
    naive_crs_spmv_cuda_launcher<IT, VT>(d_A_n_rows, d_A_col, d_A_row_ptr,
                                         d_A_val, d_x + x_offset,
                                         d_y + y_offset);
#endif

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_apply_cuda_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cuda_core(Timers *timers, KernelContext *k_ctx, Args *args,
                       Flags *flags, int A_offset, int x_offset, int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_finalize_cuda_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
#ifndef USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    int A_n_rows = args->A->n_rows;
    VT *d_y = as<VT *>(args->d_y->val);

    transfer_DtoH<VT>(d_y, args->y->val, A_n_rows);

    IF_SMAX_TIME(timers->get("finalize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_finalize_cuda_core"));
    return 0;
}

} // namespace SMAX::KERNELS::SPMV::SPMV_CUDA
