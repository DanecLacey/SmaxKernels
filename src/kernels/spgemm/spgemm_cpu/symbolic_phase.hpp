#ifndef SYMBOLIC_PHASE_HPP
#define SYMBOLIC_PHASE_HPP

#include "../spgemm_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {
namespace SPGEMM_CPU {

template <typename IT, typename VT>
int symbolic_phase_cpu(SMAX::KernelContext context, SparseMatrix *_A,
                       SparseMatrix *_B, SparseMatrix *_C) {
    IF_DEBUG(ErrorHandler::log("Entering symbolic_phase_cpu"));

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

    // Enforce dimensions of C
    C_n_rows = A_n_rows;
    C_n_cols = B_n_cols;
    C_nnz = 0;

    // Count nnz in each row of B
    IT *B_nnz_per_row = new IT[B_n_rows];
#pragma omp parallel
    {
#pragma omp for
        for (IT i = 0; i < B_n_rows; ++i) {
            B_nnz_per_row[i] = 0;
        }
#pragma omp for
        for (IT i = 0; i < B_n_rows; ++i) {
            for (IT j = B_row_ptr[i]; j < B_row_ptr[i + 1]; ++j) {
                ++B_nnz_per_row[i];
            }
        }
    }

    // Collect upper-bound offsets for each thread
    GET_THREAD_COUNT(IT, num_threads)
    IT *tl_ub = new IT[num_threads];
    IT *tl_offsets = new IT[num_threads];
    for (IT tid = 0; tid < num_threads; ++tid) {
        tl_ub[tid] = 0;
        tl_offsets[tid] = 0;
    }
#pragma omp parallel for
    for (IT i = 0; i < A_n_rows; ++i) {
        IT upper_C_row_size = 0;
        GET_THREAD_ID(IT, tid)
        for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
            IT col = A_col[j];
            upper_C_row_size += B_nnz_per_row[col];
        }
        tl_ub[tid] += upper_C_row_size;
    }

    for (IT tid = 0; tid < num_threads; ++tid) {
        tl_offsets[tid + 1] = tl_offsets[tid + 1] + tl_ub[tid];
    }

    // Allocate padded CRS arrays for C
    IT upper_nnz_bound = tl_offsets[num_threads];
    IT *padded_C_col = new IT[upper_nnz_bound];
    VT *padded_C_val = new VT[upper_nnz_bound];
    bool **used_cols = new bool *[num_threads];
    IT *tl_nnz = new IT[num_threads];
#pragma omp parallel for
    for (IT tid = 0; tid < num_threads; ++tid) {
        tl_nnz[tid] = 0;
        used_cols[tid] = new bool[upper_nnz_bound];
        for (IT i = 0; i < upper_nnz_bound; ++i) {
            used_cols[tid][i] = false;
        }
    }
    IT *C_nnz_per_row = new IT[C_n_rows];
#pragma omp parallel for
    for (IT i = 0; i < C_n_rows; ++i) {
        C_nnz_per_row[i] = 0;
    }

// Padded Gustavson's algorithm (symbolic)
#pragma omp parallel
    {
        GET_THREAD_ID(IT, tid)
        IT offset = tl_offsets[tid];
        IT tl_previous_nnz = tl_nnz[tid];
        bool *tl_used_cols = used_cols[tid];
#pragma omp for schedule(static)
        for (IT i = 0; i < A_n_rows; ++i) {
            for (IT j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
                IT left_col = A_col[j];
                for (IT k = B_row_ptr[left_col]; k < B_row_ptr[left_col + 1];
                     ++k) {
                    IT right_col = B_col[k];
                    if (!tl_used_cols[right_col]) {
                        tl_used_cols[right_col] = true;
                        padded_C_col[offset + tl_nnz[tid]] = right_col;
                        ++tl_nnz[tid];
                        ++C_nnz_per_row[i];
                    }
                }
            }
            // Reset used_cols for next row
            if (tl_nnz[tid] > tl_previous_nnz) {
                for (IT j = tl_previous_nnz; j < tl_nnz[tid]; ++j) {
                    tl_used_cols[padded_C_col[offset + j]] = false;
                }
            }
        }
    }

    // Build C_row_ptr
    // NOTE: This memory is never deleted by SMAX
    C_row_ptr = new IT[C_n_rows + 1];
    C_row_ptr[0] = 0;
    // NOTE: Does this need to be parallelized?
    for (IT i = 0; i < C_n_rows + 1; ++i) {
        C_row_ptr[i + 1] = C_row_ptr[i] + C_nnz_per_row[i];
    }

    // Compute nnz displacement array
    IT *C_nnz_displacement = new IT[num_threads];
    C_nnz_displacement[0] = 0;
    for (IT tid = 1; tid < num_threads; ++tid) {
        C_nnz_displacement[tid + 1] = C_nnz_displacement[tid + 1] + tl_nnz[tid];
    }

    // Allocate the rest of C
    C_nnz = C_row_ptr[C_n_rows];
    C_col = new IT[C_nnz];
    C_val = new VT[C_nnz];

// Compress padded_C_col to C_col
#pragma omp parallel
    {
        GET_THREAD_ID(IT, tid)
        IT offset = C_nnz_displacement[tid];
        for (IT i = 0; i < tl_nnz[tid]; ++i) {
            C_col[C_nnz_displacement[tid] + i] = padded_C_col[offset + i];
        }
    }

    IF_DEBUG(ErrorHandler::log("Exiting symbolic_phase_cpu"));
    return 0;
}

} // namespace SPGEMM_CPU
} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // SYMBOLIC_PHASE_HPP
