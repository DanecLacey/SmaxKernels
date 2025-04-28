#ifndef SPTSV_CPU_CORE_HPP
#define SPTSV_CPU_CORE_HPP

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../sptsv_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTSV {
namespace SPTSV_CPU {

template <typename IT, typename VT>
int sptsv_initialize_cpu_core(SMAX::KernelContext context, SparseMatrix *A,
                              DenseMatrix *x, DenseMatrix *y) {
    IF_DEBUG(ErrorHandler::log("Entering sptsv_initialize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting sptsv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int sptsv_apply_cpu_core(SMAX::KernelContext context, SparseMatrix *_A,
                         DenseMatrix *_X, DenseMatrix *_Y) {
    IF_DEBUG(ErrorHandler::log("Entering sptsv_apply_cpu_core"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    IT A_n_rows = as<IT>(_A->n_rows);
    IT A_n_cols = as<IT>(_A->n_cols);
    IT A_nnz = as<IT>(_A->nnz);
    IT *A_col = as<IT *>(_A->col);
    IT *A_row_ptr = as<IT *>(_A->row_ptr);
    VT *A_val = as<VT *>(_A->val);
    VT *X = as<VT *>(_X->val);
    VT *Y = as<VT *>(_Y->val);

    for (IT i = 0; i < A_n_rows; ++i) {
        VT sum = 0.0;
        VT diag = 0.0;

        for (IT idx = A_row_ptr[i]; idx < A_row_ptr[i + 1]; ++idx) {
            IT j = A_col[idx];
            VT val = A_val[idx];

            if (j < i) {
                sum += val * X[j];
            } else if (j == i) {
                diag = val;
            } else {
                IF_DEBUG(SPTSVKernelErrorHandler::super_diag());
            }
        }

        IF_DEBUG(
            if (abs(diag) < 1e-16) { SPTSVKernelErrorHandler::zero_diag(); });

        X[i] = (Y[i] - sum) / diag;
    }

    IF_DEBUG(ErrorHandler::log("Exiting sptsv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int sptsv_finalize_cpu_core(SMAX::KernelContext context, SparseMatrix *A,
                            DenseMatrix *x, DenseMatrix *y) {
    IF_DEBUG(ErrorHandler::log("Entering sptsv_finalize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting sptsv_finalize_cpu_core"));
    return 0;
}

} // namespace SPTSV_CPU
} // namespace SPTSV
} // namespace KERNELS
} // namespace SMAX

#endif // SPTSV_CPU_CORE_HPP
