#ifndef SPMV_CPU_CORE_HPP
#define SPMV_CPU_CORE_HPP

#include "../../../common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMV {
namespace SPMV_CPU {

template <typename IT, typename VT>
int spmv_initialize_cpu_core(SMAX::KernelContext context, SparseMatrix *A,
                             DenseMatrix *x, DenseMatrix *y) {
    IF_DEBUG(ErrorHandler::log("Entering spmv_initialize_cpu_core"));
    // TODO
    IF_DEBUG(ErrorHandler::log("Exiting spmv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int spmv_apply_cpu_core(SMAX::KernelContext context, SparseMatrix *_A,
                        DenseMatrix *_X, DenseMatrix *_Y) {
    IF_DEBUG(ErrorHandler::log("Entering spmv_apply_cpu_core"));

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

    IT block_vector_size = as<IT>(_X->n_cols);

    // TODO: Pretty ugly way to do this
    if (block_vector_size > 1) {
// Assuming colwise layout for now
#pragma omp for schedule(static)
        for (IT row = 0; row < A_n_rows; ++row) {
            VT tmp[block_vector_size];

            for (IT vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
                tmp[vec_idx] = VT{};
            }

            for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j) {
#pragma omp simd
                for (IT vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
                    tmp[vec_idx] +=
                        A_val[j] * X[(A_n_rows * vec_idx) + A_col[j]];
                }
            }

            for (IT vec_idx = 0; vec_idx < block_vector_size; ++vec_idx) {
                Y[row + (vec_idx * A_n_rows)] = tmp[vec_idx];
            }
        }
    } else {
#pragma omp parallel for schedule(static)
        for (IT row = 0; row < A_n_rows; ++row) {
            VT sum{};

#pragma omp simd
            for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j) {

                IF_DEBUG(
#if DEBUG_LEVEL == 3
                    printf("A_val[j] = %f\n", A_val[j]);
                    printf("A_col[j] = %d\n", A_col[j]);
                    printf("X[A_col[j]] = %f\n", X[A_col[j]]);
#endif
                );

                sum += A_val[j] * X[A_col[j]];
            }
            Y[row] = sum;
        }
    }

    IF_DEBUG(ErrorHandler::log("Exiting spmv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int spmv_finalize_cpu_core(SMAX::KernelContext context, SparseMatrix *A,
                           DenseMatrix *x, DenseMatrix *y) {
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
