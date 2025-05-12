#ifndef SMAX_SPTRSV_CPU_CORE_HPP
#define SMAX_SPTRSV_CPU_CORE_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

// Implementation files
#include "sptrsv_cpu_crs_helpers_impl.hpp"
#include "sptrsv_cpu_crs_impl.hpp"
#include "sptrsv_lvl_cpu_crs_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSV {
namespace SPTRSV_CPU {

template <typename IT, typename VT>
int sptrsv_initialize_cpu_core(KernelContext context, Args *args,
                               Flags *flags) {

    IF_DEBUG(ErrorHandler::log("Entering sptrsv_initialize_cpu_core"));

    if (!flags->diag_collected) {
        // Cast void pointers to the correct types with "as"
        // Dereference to get usable data
        int A_n_rows = args->A->n_rows;
        int A_n_cols = args->A->n_cols;
        int A_nnz = args->A->nnz;
        IT *A_col = as<IT *>(args->A->col);
        IT *A_row_ptr = as<IT *>(args->A->row_ptr);
        VT *A_val = as<VT *>(args->A->val);

        int &D_n_rows = args->D->n_rows;
        int &D_n_cols = args->D->n_cols;
        VT *&D_val = as_ptr_ref<VT>(args->D->val);
        D_n_rows = A_n_rows;
        D_n_cols = 1;
        D_val = new VT[D_n_rows];

        peel_diag_crs<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                              A_val, D_val);

        flags->diag_collected = true;
    }

    IF_DEBUG(ErrorHandler::log("Exiting sptrsv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int sptrsv_apply_cpu_core(KernelContext context, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering sptrsv_apply_cpu_core"));

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
    VT *D_val = as<VT *>(args->D->val);

    if (flags->mat_permuted) {
        int *lvl_ptr = args->uc->lvl_ptr;
        int n_levels = args->uc->n_levels;
        if (flags->mat_upper_triang) {
            crs_sputrsv_lvl<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                                    A_val, D_val, x, y, lvl_ptr, n_levels);
        } else {
            crs_spltrsv_lvl<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                                    A_val, D_val, x, y, lvl_ptr, n_levels);
        }

    } else {
        // Lower triangular matrix is the default case
        if (flags->mat_upper_triang) {
            naive_crs_sputrsv<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col,
                                      A_row_ptr, A_val, D_val, x, y);
        } else {
            // Unpermuted matrix (e.g. no lvl-set sched) is the default case
            naive_crs_spltrsv<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col,
                                      A_row_ptr, A_val, D_val, x, y);
        }
    }

    IF_DEBUG(ErrorHandler::log("Exiting sptrsv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int sptrsv_finalize_cpu_core(KernelContext context, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering sptrsv_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting sptrsv_finalize_cpu_core"));
    return 0;
}

} // namespace SPTRSV_CPU
} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_CPU_CORE_HPP
