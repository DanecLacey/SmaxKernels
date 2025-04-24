#ifndef SPMV_CPU_CORE_HPP
#define SPMV_CPU_CORE_HPP

#include "../../../common.hpp"

namespace SMAX
{

    template <typename IT, typename VT>
    int spmv_initialize_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseVector *x,
        DenseVector *y)
    {
        // TODO
        return 0;
    };

    template <typename IT, typename VT>
    int spmv_apply_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *_A,
        DenseVector *_x,
        DenseVector *_y)
    {
        // Cast void pointers to the correct types with "as"
        // Dereference to get usable data
        IT A_n_rows = as<IT>(_A->n_rows);
        IT A_n_cols = as<IT>(_A->n_cols);
        IT A_nnz = as<IT>(_A->nnz);
        IT *A_col = as<IT *>(_A->col);
        IT *A_row_ptr = as<IT *>(_A->row_ptr);
        VT *A_val = as<VT *>(_A->val);
        VT *x = as<VT *>(_x->val);
        VT *y = as<VT *>(_y->val);

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (IT row = 0; row < A_n_rows; ++row)
            {
                VT sum{};

#pragma omp simd
                for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j)
                {
                    sum += A_val[j] * x[A_col[j]];
// #ifdef DEBUG_MODE_FINE
//                 printf("rank %i: %f += %f * %f using col idx %i w/ j=%i, row=%i\n", *my_rank, sum, values[j], x[col_idxs[j]], col_idxs[j], j, row);
// #endif
                }
                y[row] = sum;
            }
        }

        return 0;
    }

    template <typename IT, typename VT>
    int spmv_finalize_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseVector *x,
        DenseVector *y)
    {
        // TODO
        return 0;
    }

} // namespace SMAX

#endif // SPMV_CPU_CORE_HPP
