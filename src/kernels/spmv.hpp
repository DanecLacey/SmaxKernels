#ifndef SPMV_HPP
#define SPMV_HPP

#include "../common.hpp"
#include "../macros.hpp"
#include "spmv/spmv_cpu.hpp"

#include <functional>
#include <cstdarg>

namespace SMAX
{

    int spmv_register_A(
        SparseMatrix *A,
        va_list args)
    {
        A->n_rows = va_arg(args, void *);
        A->n_cols = va_arg(args, void *);
        A->nnz = va_arg(args, void *);
        A->col = va_arg(args, void **);
        A->row_ptr = va_arg(args, void **);
        A->val = va_arg(args, void **);

        return 0;
    }

    int spmv_register_B(
        DenseMatrix *X,
        va_list args)
    {
        X->n_rows = va_arg(args, void *);
        X->n_cols = va_arg(args, void *);
        X->val = va_arg(args, void **);

        return 0;
    }

    int spmv_register_C(
        DenseMatrix *Y,
        va_list args)
    {
        Y->n_rows = va_arg(args, void *);
        Y->n_cols = va_arg(args, void *);
        Y->val = va_arg(args, void **);

        return 0;
    }

    int spmv_dispatch(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseMatrix *X,
        DenseMatrix *Y,
        std::function<int(SMAX::KernelContext, SparseMatrix *, DenseMatrix *, DenseMatrix *)> cpu_func,
        const char *label)
    {
        switch (context.platform_type)
        {
        case SMAX::CPU:
            CHECK_ERROR(cpu_func(context, A, X, Y), label);
            break;
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
        return 0;
    }

    int spmv_initialize(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseMatrix *X,
        DenseMatrix *Y)
    {
        return spmv_dispatch(
            context, A, X, Y,
            [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y)
            { return spmv_initialize_cpu(context, A, X, Y); },
            "spmv_initialize");
    }

    int spmv_apply(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseMatrix *X,
        DenseMatrix *Y)
    {
        return spmv_dispatch(
            context, A, X, Y,
            [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y)
            { return spmv_apply_cpu(context, A, X, Y); },
            "spmv_apply");
    }

    int spmv_finalize(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseMatrix *X,
        DenseMatrix *Y)
    {
        return spmv_dispatch(
            context, A, X, Y,
            [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y)
            { return spmv_finalize_cpu(context, A, X, Y); },
            "spmv_finalize");
    }

} // namespace SMAX

#endif // SPMV_HPP
