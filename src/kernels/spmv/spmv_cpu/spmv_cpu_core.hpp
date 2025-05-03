#ifndef SPMV_CPU_CORE_HPP
#define SPMV_CPU_CORE_HPP

#include "../../../common.hpp"
#include "spmv_cpu_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMV {
namespace SPMV_CPU {

template <typename IT, typename VT>
int spmv_initialize_cpu_core(KernelContext context, SparseMatrix *_A,
                             DenseMatrix *_X, DenseMatrix *_Y, int A_offset,
                             int X_offset, int Y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmv_initialize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spmv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spmv_apply_cpu_core(KernelContext context, SparseMatrix *_A,
                        DenseMatrix *_X, DenseMatrix *_Y, int A_offset,
                        int X_offset, int Y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmv_apply_cpu_core"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = _A->n_rows;
    int A_n_cols = _A->n_cols;
    int A_nnz = _A->nnz;
    IT *A_col = as<IT *>(_A->col);
    IT *A_row_ptr = as<IT *>(_A->row_ptr);
    VT *A_val = as<VT *>(_A->val);
    VT *X = as<VT *>(_X->val);
    VT *Y = as<VT *>(_Y->val);
    int block_vector_size = _X->n_cols;

#if 1
    basic_spmv<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val,
                       X + X_offset, Y_offset, block_vector_size);
#endif

    IF_DEBUG(ErrorHandler::log("Exiting spmv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int spmv_finalize_cpu_core(KernelContext context, SparseMatrix *A,
                           DenseMatrix *X, DenseMatrix *Y, int A_offset,
                           int X_offset, int Y_offset) {
    IF_DEBUG(ErrorHandler::log("Entering spmv_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spmv_finalize_cpu_core"));
    return 0;
}

} // namespace SPMV_CPU
} // namespace SPMV
} // namespace KERNELS
} // namespace SMAX

#endif // SPMV_CPU_CORE_HPP
