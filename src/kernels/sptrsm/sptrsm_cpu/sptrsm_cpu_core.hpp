#ifndef SMAX_SPTRSM_CPU_CORE_HPP
#define SMAX_SPTRSM_CPU_CORE_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsm_common.hpp"
#include "sptrsm_cpu_crs_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSM {
namespace SPTRSM_CPU {

template <typename IT, typename VT>
int sptrsm_initialize_cpu_core(KernelContext context, Args *args,
                               Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering sptrsm_initialize_cpu_core"));
    // TODO
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
    int block_vector_size = args->X->n_cols;

#if 1
    native_crs_sptrsm<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr,
                              A_val, X, Y, block_vector_size);
#endif

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
