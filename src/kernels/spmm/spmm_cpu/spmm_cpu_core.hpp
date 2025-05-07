#ifndef SMAX_SPMM_CPU_CORE_HPP
#define SMAX_SPMM_CPU_CORE_HPP

#include "../../../common.hpp"
#include "spmm_cpu_crs_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMM {
namespace SPMM_CPU {

template <typename IT, typename VT>
int spmm_initialize_cpu_core(KernelContext context, Args *args, Flags *flags,
                             int A_offset, int X_offset, int Y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmm_initialize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spmm_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spmm_apply_cpu_core(KernelContext context, Args *args, Flags *flags,
                        int A_offset, int X_offset, int Y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmm_apply_cpu_core"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = args->A->n_rows;
    int A_n_cols = args->A->n_cols;
    int A_nnz = args->A->nnz;
    IT *A_col = as<IT *>(args->A->col);
    IT *A_row_ptr = as<IT *>(args->A->row_ptr);
    VT *A_val = as<VT *>(args->A->val);
    VT *X = as<VT *>(args->X->val);
    VT *Y = as<VT *>(args->Y->val);
    int block_vector_size = args->X->n_cols;

#if 1
    naive_crs_spmm<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val,
                           X + X_offset, Y + Y_offset, block_vector_size);
#endif

    IF_DEBUG(ErrorHandler::log("Exiting spmm_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int spmm_finalize_cpu_core(KernelContext context, Args *args, Flags *flags,
                           int A_offset, int X_offset, int Y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmm_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spmm_finalize_cpu_core"));
    return 0;
}

} // namespace SPMM_CPU
} // namespace SPMM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMM_CPU_CORE_HPP
