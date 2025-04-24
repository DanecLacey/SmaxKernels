#ifndef SPMV_CPU_CORE_HPP
#define SPMV_CPU_CORE_HPP

#include "../../../common.hpp"

namespace SMAX
{

    template <typename IT, typename VT>
    int spmv_initialize_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseMatrix *x,
        DenseMatrix *y)
    {
        // TODO
        return 0;
    };

    template <typename IT, typename VT>
    int spmv_apply_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *_A,
        DenseMatrix *_X,
        DenseMatrix *_Y)
    {
        // Cast void pointers to the correct types with "as"
        // Dereference to get usable data
        int A_n_rows = as<int>(_A->n_rows);
        int A_n_cols = as<int>(_A->n_cols);
        int A_nnz = as<int>(_A->nnz);
        IT *A_col = as<IT *>(_A->col);
        IT *A_row_ptr = as<IT *>(_A->row_ptr);
        VT *A_val = as<VT *>(_A->val);
        VT *X = as<VT *>(_X->val);
        VT *Y = as<VT *>(_Y->val);

        int block_vector_size = as<int>(_X->n_cols);

        // TODO: Pretty ugly way to do this
        if (block_vector_size > 1)
        {
// Assuming colwise layout for now
#pragma omp for schedule(static)
            for (int row = 0; row < A_n_rows; ++row)
            {
                VT tmp[block_vector_size];

                for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx)
                {
                    tmp[vec_idx] = VT{};
                }

                for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j)
                {
#pragma omp simd
                    for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx)
                    {
                        tmp[vec_idx] += A_val[j] * X[(A_n_rows * vec_idx) + A_col[j]];
                    }
                }

                for (int vec_idx = 0; vec_idx < block_vector_size; ++vec_idx)
                {
                    Y[row + (vec_idx * A_n_rows)] = tmp[vec_idx];
                }
            }
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (IT row = 0; row < A_n_rows; ++row)
            {
                VT sum{};

#pragma omp simd
                for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j)
                {
                    sum += A_val[j] * X[A_col[j]];
                }
                Y[row] = sum;
            }
        }

        return 0;
    }

    template <typename IT, typename VT>
    int spmv_finalize_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseMatrix *x,
        DenseMatrix *y)
    {
        // TODO
        return 0;
    }

} // namespace SMAX

#endif // SPMV_CPU_CORE_HPP
