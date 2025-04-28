#ifndef SPTSV_HPP
#define SPTSV_HPP

#include "../common.hpp"
#include "../macros.hpp"
#include "sptsv/sptsv_cpu.hpp"

#include <cstdarg>
#include <functional>

namespace SMAX {
namespace KERNELS {

int sptsv_register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, void *);
    A->n_cols = va_arg(args, void *);
    A->nnz = va_arg(args, void *);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int sptsv_register_B(DenseMatrix *X, va_list args) {
    X->n_rows = va_arg(args, void *);
    X->n_cols = va_arg(args, void *);
    X->val = va_arg(args, void **);

    return 0;
}

int sptsv_register_C(DenseMatrix *Y, va_list args) {
    Y->n_rows = va_arg(args, void *);
    Y->n_cols = va_arg(args, void *);
    Y->val = va_arg(args, void **);

    return 0;
}

int sptsv_dispatch(SMAX::KernelContext context, SparseMatrix *A, DenseMatrix *X,
                   DenseMatrix *Y,
                   std::function<int(SMAX::KernelContext, SparseMatrix *,
                                     DenseMatrix *, DenseMatrix *)>
                       cpu_func,
                   const char *label) {
    switch (context.platform_type) {
    case SMAX::CPU:
        CHECK_ERROR(cpu_func(context, A, X, Y), label);
        break;
    default:
        std::cerr << "Error: Platform not supported\n";
        return 1;
    }
    return 0;
}

int sptsv_initialize(SMAX::KernelContext context, SparseMatrix *A,
                     DenseMatrix *X, DenseMatrix *Y) {
    return sptsv_dispatch(
        context, A, X, Y,
        [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y) {
            return SPTSV::sptsv_initialize_cpu(context, A, X, Y);
        },
        "sptsv_initialize");
}

int sptsv_apply(SMAX::KernelContext context, SparseMatrix *A, DenseMatrix *X,
                DenseMatrix *Y) {
    return sptsv_dispatch(
        context, A, X, Y,
        [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y) {
            return SPTSV::sptsv_apply_cpu(context, A, X, Y);
        },
        "sptsv_apply");
}

int sptsv_finalize(SMAX::KernelContext context, SparseMatrix *A, DenseMatrix *X,
                   DenseMatrix *Y) {
    return sptsv_dispatch(
        context, A, X, Y,
        [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y) {
            return SPTSV::sptsv_finalize_cpu(context, A, X, Y);
        },
        "sptsv_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SPTSV_HPP
