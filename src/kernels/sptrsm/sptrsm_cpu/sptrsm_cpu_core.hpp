#ifndef SMAX_SPTRSM_CPU_CORE_HPP
#define SMAX_SPTRSM_CPU_CORE_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsm_common.hpp"

// Implementation files
#include "sptrsm_cpu_crs_helpers_impl.hpp"
#include "sptrsm_cpu_crs_impl.hpp"
// #include "sptrsm_lvl_cpu_crs_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSM {
namespace SPTRSM_CPU {

template <typename IT, typename VT>
int sptrsm_initialize_cpu_core(KernelContext context, Args *args,
                               Flags *flags) {

    IF_DEBUG(ErrorHandler::log("Entering sptrsm_initialize_cpu_core"));

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

    IF_DEBUG(ErrorHandler::log("Exiting sptrsm_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int sptrsm_apply_cpu_core(KernelContext context, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering sptrsm_apply_cpu_core"));

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
    VT *D_val = as<VT *>(args->D->val);
    int block_vector_size = args->X->n_cols;

    // if (flags->mat_permuted) {
    //     int *lvl_ptr = args->uc->lvl_ptr;
    //     int n_levels = args->uc->n_levels;
    //     if (flags->mat_upper_triang) {
    //         crs_sputrsm_lvl<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col,
    //         A_row_ptr,
    //                                 A_val, D_val, X, Y, lvl_ptr, n_levels);
    //     } else {
    //         crs_spltrsm_lvl<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col,
    //         A_row_ptr,
    //                                 A_val, D_val, X, Y, lvl_ptr, n_levels);
    //     }

    // } else {
    // Lower triangular matrix is the default case
    if (flags->mat_upper_triang) {
        naive_crs_sputrsm<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                                  A_val, D_val, X, Y, block_vector_size);
    } else {
        // Unpermuted matrix (e.g. no lvl-set sched) is the default case
        naive_crs_spltrsm<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                                  A_val, D_val, X, Y, block_vector_size);
    }
    // }

    IF_DEBUG(ErrorHandler::log("Exiting sptrsm_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int sptrsm_finalize_cpu_core(KernelContext context, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering sptrsm_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting sptrsm_finalize_cpu_core"));
    return 0;
}

} // namespace SPTRSM_CPU
} // namespace SPTRSM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSM_CPU_CORE_HPP
