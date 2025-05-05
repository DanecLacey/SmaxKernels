#ifndef SMAX_SPTRSV_CPU_CORE_HPP
#define SMAX_SPTRSV_CPU_CORE_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptrsv_common.hpp"
#include "sptrsv_cpu_crs_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSV {
namespace SPTRSV_CPU {

template <typename IT, typename VT>
int sptrsv_initialize_cpu_core(KernelContext context, SparseMatrix *A,
                               DenseMatrix *x, DenseMatrix *y) {
    IF_DEBUG(ErrorHandler::log("Entering sptrsv_initialize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting sptrsv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int sptrsv_apply_cpu_core(KernelContext context, SparseMatrix *_A,
                          DenseMatrix *_X, DenseMatrix *_Y) {
    IF_DEBUG(ErrorHandler::log("Entering sptrsv_apply_cpu_core"));

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

#if 1
    naive_crs_sptrsv<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val,
                             X, Y);
#elif 0
    // spltsv_lvl();
#elif 0
    // spltsv_2stage();
#elif 0
    // spltsv_mc();
#endif

    IF_DEBUG(ErrorHandler::log("Exiting sptrsv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int sptrsv_finalize_cpu_core(KernelContext context, SparseMatrix *A,
                             DenseMatrix *x, DenseMatrix *y) {
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
