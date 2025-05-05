#ifndef SMAX_SPGEMV_CPU_CORE_HPP
#define SMAX_SPGEMV_CPU_CORE_HPP

#include "../../../common.hpp"
#include "spgemv_cpu_crs_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMV {
namespace SPGEMV_CPU {

template <typename IT, typename VT>
int spgemv_initialize_cpu_core(KernelContext context, SparseMatrix *_A,
                               SparseVector *_spX, SparseVectorRef *_spY_ref) {
    IF_DEBUG(ErrorHandler::log("Entering spgemv_initialize_cpu_core"));
    // TODO

    IF_DEBUG(ErrorHandler::log("Exiting spgemv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spgemv_apply_cpu_core(KernelContext context, SparseMatrix *_A,
                          SparseVector *_spX, SparseVectorRef *_spY_ref) {

    IF_DEBUG(ErrorHandler::log("Entering spgemv_apply_cpu_core"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = _A->n_rows;
    int A_n_cols = _A->n_cols;
    int A_nnz = _A->nnz;
    IT *A_col = as<IT *>(_A->col);
    IT *A_row_ptr = as<IT *>(_A->row_ptr);
    VT *A_val = as<VT *>(_A->val);
    int spX_n_rows = _spX->n_rows;
    int spX_nnz = _spX->nnz;
    IT *spX_idx = as<IT *>(_spX->idx);
    VT *spX_val = as<VT *>(_spX->val);
    int &spY_n_rows = *(_spY_ref->n_rows);
    int &spY_nnz = *(_spY_ref->nnz);
    IT *&spY_idx = as<IT *>(_spY_ref->idx);
    VT *&spY_val = as<VT *>(_spY_ref->val);

#if 1
    naive_crs_coo_spgemv(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val,
                         spX_n_rows, spX_nnz, spX_idx, spX_val, spY_n_rows,
                         spY_nnz, spY_idx, spY_val);
#endif

    IF_DEBUG(ErrorHandler::log("Exiting spgemv_apply_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spgemv_finalize_cpu_core(KernelContext context, SparseMatrix *_A,
                             SparseVector *_spX, SparseVectorRef *_spY_ref) {
    IF_DEBUG(ErrorHandler::log("Entering spgemv_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spgemv_finalize_cpu_core"));
    return 0;
};

} // namespace SPGEMV_CPU
} // namespace SPGEMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMV_CPU_CORE_HPP
