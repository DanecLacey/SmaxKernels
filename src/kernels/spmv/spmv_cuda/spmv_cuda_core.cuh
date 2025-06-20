#pragma once

#include "../../../common.hpp"
#include "../../../platforms/cuda/cuda.cuh"
#include "spmv_cuda_crs_impl.cuh"
#include "spmv_cuda_scs_impl.cuh"

namespace SMAX::KERNELS::SPMV::CUDA {

template <typename IT, typename VT>
int initialize_cuda_core(Timers *timers, KernelContext *k_ctx, Args *args,
                         Flags *flags, ULL A_offset, ULL x_offset,
                         ULL y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_initialize_cuda_core"));
    IF_SMAX_TIME(timers->get("initialize")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    ULL vector_size;

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    if (flags->is_mat_scs) {
        ULL A_n_elements = args->A->scs->n_elements;
        ULL A_n_rows_padded = args->A->scs->n_rows_padded;
        ULL A_n_chunks = args->A->scs->n_chunks;
        IT *A_chunk_ptr = as<IT *>(args->A->scs->chunk_ptr);
        IT *A_chunk_lengths = as<IT *>(args->A->scs->chunk_lengths);
        IT *A_col = as<IT *>(args->A->scs->col);
        VT *A_val = as<VT *>(args->A->scs->val);

        // Copy typed pointers from host to device
        transfer_HtoD<IT>(A_chunk_ptr, args->d_A->scs->chunk_ptr,
                          A_n_chunks + 1);
        transfer_HtoD<IT>(A_chunk_lengths, args->d_A->scs->chunk_lengths,
                          A_n_chunks);
        transfer_HtoD<IT>(A_col, args->d_A->scs->col, A_n_elements);
        transfer_HtoD<VT>(A_val, args->d_A->scs->val, A_n_elements);

        // Copy metadata from host matrix
        args->d_A->scs->n_chunks = args->A->scs->n_chunks;
        args->d_A->scs->C = args->A->scs->C;
        vector_size = A_n_rows_padded;
    } else {
        ULL A_n_rows = args->A->crs->n_rows;
        ULL A_nnz = args->A->crs->nnz;
        IT *A_col = as<IT *>(args->A->crs->col);
        IT *A_row_ptr = as<IT *>(args->A->crs->row_ptr);
        VT *A_val = as<VT *>(args->A->crs->val);
        vector_size = A_n_rows;

        // Copy typed pointers from host to device
        transfer_HtoD<IT>(A_col, args->d_A->crs->col, A_nnz);
        transfer_HtoD<IT>(A_row_ptr, args->d_A->crs->row_ptr, A_n_rows + 1);
        transfer_HtoD<VT>(A_val, args->d_A->crs->val, A_nnz);

        // Copy metadata from host matrix
        args->d_A->crs->n_rows = args->A->crs->n_rows;
        args->d_A->crs->n_cols = args->A->crs->n_cols;
        args->d_A->crs->nnz = args->A->crs->nnz;
    }

    VT *x = as<VT *>(args->x->val);
    VT *y = as<VT *>(args->y->val);
    transfer_HtoD<VT>(x, args->d_x->val, vector_size);
    transfer_HtoD<VT>(y, args->d_y->val, vector_size);

    IF_SMAX_TIME(timers->get("initialize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_initialize_cuda_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cuda_core(Timers *timers, KernelContext *k_ctx, Args *args,
                    Flags *flags, ULL A_offset, ULL x_offset, ULL y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_apply_cuda_core"));
    IF_SMAX_TIME(timers->get("apply")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)flags;
    (void)A_offset;

    // clang-format off
    if (flags->is_mat_scs) {
        naive_scs_spmv_cuda_launcher<IT, VT>(
            args->d_A->scs->C,
            args->d_A->scs->n_chunks,
            as<IT *>(args->d_A->scs->chunk_ptr),
            as<IT *>(args->d_A->scs->chunk_lengths),
            as<IT *>(args->d_A->scs->col),
            as<VT *>(args->d_A->scs->val),
            as<VT *>(args->d_x->val) + x_offset,
            as<VT *>(args->d_y->val) + y_offset);
    }
    else{
        naive_crs_spmv_cuda_launcher<IT, VT>(
            args->d_A->crs->n_rows, 
            as<IT *>(args->d_A->crs->col), 
            as<IT *>(args->d_A->crs->row_ptr),
            as<VT *>(args->d_A->crs->val),
            as<VT *>(args->d_x->val) + x_offset,
            as<VT *>(args->d_y->val) + y_offset);
    }
    // clang-format on

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_apply_cuda_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cuda_core(Timers *timers, KernelContext *k_ctx, Args *args,
                       Flags *flags, ULL A_offset, ULL x_offset, ULL y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_finalize_cuda_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
#if !SMAX_USE_TIMERS
    (void)timers;
#endif
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    ULL vector_size;
    if (flags->is_mat_scs) {
        ULL A_n_rows_padded = args->A->scs->n_rows_padded;
        vector_size = A_n_rows_padded;
    } else {
        ULL A_n_rows = args->A->crs->n_rows;
        vector_size = A_n_rows;
    }

    VT *d_y = as<VT *>(args->d_y->val);

    transfer_DtoH<VT>(d_y, args->y->val, vector_size);

    IF_SMAX_TIME(timers->get("finalize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_finalize_cuda_core"));
    return 0;
}

} // namespace SMAX::KERNELS::SPMV::CUDA
