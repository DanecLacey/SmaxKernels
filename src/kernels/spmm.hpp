#ifndef SMAX_SPMM_HPP
#define SMAX_SPMM_HPP

#include "../common.hpp"
#include "../macros.hpp"
#include "spmm/spmm_cpu.hpp"

#include <cstdarg>
#include <functional>

namespace SMAX {
namespace KERNELS {

int spmm_register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, int);
    A->n_cols = va_arg(args, int);
    A->nnz = va_arg(args, int);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int spmm_register_B(DenseMatrix *X, va_list args) {
    X->n_rows = va_arg(args, int);
    X->n_cols = va_arg(args, int);
    X->val = va_arg(args, void **);

    return 0;
}

int spmm_register_C(DenseMatrix *Y, va_list args) {
    Y->n_rows = va_arg(args, int);
    Y->n_cols = va_arg(args, int);
    Y->val = va_arg(args, void **);

    return 0;
}

int spmm_dispatch(
    KernelContext context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y,
    int A_offset, int X_offset, int Y_offset,
    std::function<int(KernelContext, SparseMatrix *, DenseMatrix *,
                      DenseMatrix *, int, int, int)>
        cpu_func,
    const char *label) {
    switch (context.platform_type) {
    case SMAX::CPU:
        CHECK_ERROR(cpu_func(context, A, X, Y, A_offset, X_offset, Y_offset),
                    label);
        break;
    default:
        std::cerr << "Error: Platform not supported\n";
        return 1;
    }
    return 0;
}

int spmm_initialize(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                    DenseMatrix *Y, int A_offset, int X_offset, int Y_offset) {

    return spmm_dispatch(
        context, A, X, Y, A_offset, X_offset, Y_offset,
        [](KernelContext context, SparseMatrix *A, DenseMatrix *X,
           DenseMatrix *Y, int A_offset, int X_offset, int Y_offset) {
            return SPMM::spmm_initialize_cpu(context, A, X, Y, A_offset,
                                             X_offset, Y_offset);
        },
        "spmm_initialize");
}

int spmm_apply(KernelContext context, SparseMatrix *A, DenseMatrix *X,
               DenseMatrix *Y, int A_offset, int X_offset, int Y_offset) {

    return spmm_dispatch(
        context, A, X, Y, A_offset, X_offset, Y_offset,
        [](KernelContext context, SparseMatrix *A, DenseMatrix *X,
           DenseMatrix *Y, int A_offset, int X_offset, int Y_offset) {
            return SPMM::spmm_apply_cpu(context, A, X, Y, A_offset, X_offset,
                                        Y_offset);
        },
        "spmm_apply");
}

int spmm_finalize(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                  DenseMatrix *Y, int A_offset, int X_offset, int Y_offset) {

    return spmm_dispatch(
        context, A, X, Y, A_offset, X_offset, Y_offset,
        [](KernelContext context, SparseMatrix *A, DenseMatrix *X,
           DenseMatrix *Y, int A_offset, int X_offset, int Y_offset) {
            return SPMM::spmm_finalize_cpu(context, A, X, Y, A_offset, X_offset,
                                           Y_offset);
        },
        "spmm_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMM_HPP
