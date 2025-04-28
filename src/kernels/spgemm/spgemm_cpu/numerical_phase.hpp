#ifndef NUMERICAL_PHASE_HPP
#define NUMERICAL_PHASE_HPP

#include "../spgemm_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {
namespace SPGEMM_CPU {

template <typename IT, typename VT>
int numerical_phase_cpu(SMAX::KernelContext context, SparseMatrix *_A,
                        SparseMatrix *_B, SparseMatrix *_C) {
    IF_DEBUG(ErrorHandler::log("Entering numerical_phase_cpu"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    IT A_n_rows = as<IT>(_A->n_rows);
    IT A_n_cols = as<IT>(_A->n_cols);
    IT A_nnz = as<IT>(_A->nnz);
    IT *A_col = as<IT *>(_A->col);
    IT *A_row_ptr = as<IT *>(_A->row_ptr);
    VT *A_val = as<VT *>(_A->val);

    IT B_n_rows = as<IT>(_B->n_rows);
    IT B_n_cols = as<IT>(_B->n_cols);
    IT B_nnz = as<IT>(_B->nnz);
    IT *B_col = as<IT *>(_B->col);
    IT *B_row_ptr = as<IT *>(_B->row_ptr);
    VT *B_val = as<VT *>(_B->val);

    // Since we want to modify the data pointed to by _C,
    // we need references to the data
    IT &C_n_rows = as<IT>(_C->n_rows);
    IT &C_n_cols = as<IT>(_C->n_cols);
    IT &C_nnz = as<IT>(_C->nnz);
    IT *&C_col = as<IT *>(_C->col);
    IT *&C_row_ptr = as<IT *>(_C->row_ptr);
    VT *&C_val = as<VT *>(_C->val);

    GET_THREAD_COUNT(IT, num_threads)

    // Prepare thread local dense accumulators
    VT **dense_accumulators = new VT *[num_threads];

    for (IT tid = 0; tid < num_threads; ++tid) {
        dense_accumulators[tid] = new VT[C_n_cols];
    }

#pragma omp parallel
    {
        GET_THREAD_ID(IT, tid)
        for (IT i = 0; i < C_n_cols; ++i) {
            dense_accumulators[tid][i] = (VT)0.0;
        }
    }

// Gustavson's algorithm (numerical)
#pragma omp parallel
    {
        GET_THREAD_ID(IT, tid)
#pragma omp for schedule(static)
        for (IT i = 0; i < A_n_rows; ++i) {
            for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
                IT left_col = A_col[j];
                for (IT k = B_row_ptr[left_col]; k < B_row_ptr[left_col + 1];
                     ++k) {
                    IT right_col = B_col[k];

                    // Accumulate intermediate results
                    dense_accumulators[tid][right_col] += A_val[j] * B_val[k];
                }
            }
            // Write row-local accumulators to C
            for (IT j = C_row_ptr[i]; j < C_row_ptr[i + 1]; ++j) {
                C_val[j] = dense_accumulators[tid][C_col[j]];
                dense_accumulators[tid][C_col[j]] = (VT)0.0;
            }
        }
    }

    IF_DEBUG(ErrorHandler::log("Exiting numerical_phase_cpu"));
    return 0;
}

} // namespace SPGEMM_CPU
} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // NUMERICAL_PHASE_HPP
