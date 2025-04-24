#ifndef SPGEMM_HPP
#define SPGEMM_HPP

#include "../common.hpp"
#include "../macros.hpp"
#include "spgemm/spgemm_cpu.hpp"

#include <functional>
#include <cstdarg>

namespace SMAX
{

    int spgemm_register_A(
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

    int spgemm_register_B(
        SparseMatrix *B,
        va_list args)
    {
        B->n_rows = va_arg(args, void *);
        B->n_cols = va_arg(args, void *);
        B->nnz = va_arg(args, void *);
        B->col = va_arg(args, void **);
        B->row_ptr = va_arg(args, void **);
        B->val = va_arg(args, void **);

        return 0;
    }

    int spgemm_register_C(
        SparseMatrix *C,
        va_list args)
    {
        C->n_rows = va_arg(args, void *);
        C->n_cols = va_arg(args, void *);
        C->nnz = va_arg(args, void *);
        C->col = va_arg(args, void **);
        C->row_ptr = va_arg(args, void **);
        C->val = va_arg(args, void **);

        return 0;
    }

    int spgemm_dispatch(
        SMAX::KernelContext context,
        SparseMatrix *A,
        SparseMatrix *B,
        SparseMatrix *C,
        std::function<int(SMAX::KernelContext, SparseMatrix *, SparseMatrix *, SparseMatrix *)> cpu_func,
        const char *label)
    {
        switch (context.platform_type)
        {
        case SMAX::CPU:
            CHECK_ERROR(cpu_func(context, A, B, C), label);
            break;
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
        return 0;
    }

    int spgemm_initialize(
        SMAX::KernelContext context,
        SparseMatrix *A,
        SparseMatrix *B,
        SparseMatrix *C)
    {
        return spgemm_dispatch(
            context, A, B, C,
            [](auto context, SparseMatrix *A, SparseMatrix *B, SparseMatrix *C)
            { return spgemm_initialize_cpu(context, A, B, C); },
            "spgemm_initialize");
    }

    int spgemm_apply(
        SMAX::KernelContext context,
        SparseMatrix *A,
        SparseMatrix *B,
        SparseMatrix *C)
    {
        return spgemm_dispatch(
            context, A, B, C,
            [](auto context, SparseMatrix *A, SparseMatrix *B, SparseMatrix *C)
            { return spgemm_apply_cpu(context, A, B, C); },
            "spgemm_apply");
    }

    int spgemm_finalize(
        SMAX::KernelContext context,
        SparseMatrix *A,
        SparseMatrix *B,
        SparseMatrix *C)
    {
        return spgemm_dispatch(
            context, A, B, C,
            [](auto context, SparseMatrix *A, SparseMatrix *B, SparseMatrix *C)
            { return spgemm_finalize_cpu(context, A, B, C); },
            "spgemm_finalize");
    }

} // namespace SMAX

#endif // SPGEMM_HPP
