#ifndef SMAX_SPGEMV_CPU_CORE_HPP
#define SMAX_SPGEMV_CPU_CORE_HPP

#include "../../../common.hpp"
#include "spgemv_cpu_crs_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMV {
namespace SPGEMV_CPU {

template <typename IT, typename VT>
int spgemv_initialize_cpu_core(KernelContext context, Args *args,
                               Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering spgemv_initialize_cpu_core"));
    // TODO

    IF_DEBUG(ErrorHandler::log("Exiting spgemv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spgemv_apply_cpu_core(KernelContext context, Args *args, Flags *flags) {

    IF_DEBUG(ErrorHandler::log("Entering spgemv_apply_cpu_core"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = args->A->n_rows;
    int A_n_cols = args->A->n_cols;
    int A_nnz = args->A->nnz;
    IT *A_col = as<IT *>(args->A->col);
    IT *A_row_ptr = as<IT *>(args->A->row_ptr);
    VT *A_val = as<VT *>(args->A->val);
    int x_n_rows = args->x->n_rows;
    int x_nnz = args->x->nnz;
    IT *x_idx = as<IT *>(args->x->idx);
    VT *x_val = as<VT *>(args->x->val);
    int &y_n_rows = *(args->y->n_rows);
    int &y_nnz = *(args->y->nnz);
    IT *&y_idx = as<IT *>(args->y->idx);
    VT *&y_val = as<VT *>(args->y->val);

#if 1
    naive_crs_coo_spgemv(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val,
                         x_n_rows, x_nnz, x_idx, x_val, y_n_rows, y_nnz, y_idx,
                         y_val);
#endif

    IF_DEBUG(ErrorHandler::log("Exiting spgemv_apply_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spgemv_finalize_cpu_core(KernelContext context, Args *args, Flags *flags) {
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
