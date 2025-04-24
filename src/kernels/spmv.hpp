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
        DenseVector *x,
        va_list args)
    {
        x->n_rows = va_arg(args, void *);
        x->val = va_arg(args, void **);

        return 0;
    }

    int spmv_register_C(
        DenseVector *y,
        va_list args)
    {
        y->n_rows = va_arg(args, void *);
        y->val = va_arg(args, void **);

        return 0;
    }

    int spmv_dispatch(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseVector *x,
        DenseVector *y,
        std::function<int(SMAX::KernelContext, SparseMatrix *, DenseVector *, DenseVector *)> cpu_func,
        const char *label)
    {
        switch (context.platform_type)
        {
        case SMAX::CPU:
            CHECK_ERROR(cpu_func(context, A, x, y), label);
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
        DenseVector *x,
        DenseVector *y)
    {
        return spmv_dispatch(
            context, A, x, y,
            [](auto context, SparseMatrix *A, DenseVector *x, DenseVector *y)
            { return spmv_initialize_cpu(context, A, x, y); },
            "spmv_initialize");
    }

    int spmv_apply(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseVector *x,
        DenseVector *y)
    {
        return spmv_dispatch(
            context, A, x, y,
            [](auto context, SparseMatrix *A, DenseVector *x, DenseVector *y)
            { return spmv_apply_cpu(context, A, x, y); },
            "spmv_apply");
    }

    int spmv_finalize(
        SMAX::KernelContext context,
        SparseMatrix *A,
        DenseVector *x,
        DenseVector *y)
    {
        return spmv_dispatch(
            context, A, x, y,
            [](auto context, SparseMatrix *A, DenseVector *x, DenseVector *y)
            { return spmv_finalize_cpu(context, A, x, y); },
            "spmv_finalize");
    }

} // namespace SMAX

#endif // SPMV_HPP
