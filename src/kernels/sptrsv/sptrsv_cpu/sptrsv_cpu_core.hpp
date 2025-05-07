#ifndef SMAx_SPTRSV_CPU_CORE_HPP
#define SMAx_SPTRSV_CPU_CORE_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"

// Implementation files
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

    if (!flags->lvl_ptr_collected) {
        int A_n_rows = args->A->n_rows;
        IT *A_row_ptr = as<IT *>(args->A->row_ptr);
        IT *A_col = as<IT *>(args->A->col);

        // NOTE: We far overallocate, since we have no idea how many level exist
        // at this point
        args->lvl_ptr = new int[A_n_rows];

#pragma omp parallel for
        for (int i = 0; i < A_n_rows; ++i) {
            args->lvl_ptr[i] = 0;
        }

        collect_lvl_ptr(A_n_rows, A_row_ptr, A_col, args->lvl_ptr,
                        args->n_levels);

        flags->lvl_ptr_collected = true;
        IF_DEBUG(ErrorHandler::log("Detected %d levels", args->n_levels));
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
    int *lvl_ptr = args->lvl_ptr;
    int n_levels = args->n_levels;

    if (flags->mat_permuted) {
        crs_sptrsv_lvl<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                               A_val, x, y, lvl_ptr, n_levels);
    } else {
        naive_crs_sptrsv<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                                 A_val, x, y);
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
