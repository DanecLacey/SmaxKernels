#ifndef SMAX_SPTRSM_HPP
#define SMAX_SPTRSM_HPP

#include "../common.hpp"
#include "../macros.hpp"
#include "sptrsm/sptrsm_cpu.hpp"

#include <cstdarg>
#include <functional>

namespace SMAX {
namespace KERNELS {

int sptrsm_register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, int);
    A->n_cols = va_arg(args, int);
    A->nnz = va_arg(args, int);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int sptrsm_register_B(DenseMatrix *X, va_list args) {
    X->n_rows = va_arg(args, int);
    X->n_cols = va_arg(args, int);
    X->val = va_arg(args, void **);

    return 0;
}

int sptrsm_register_C(DenseMatrix *Y, va_list args) {
    Y->n_rows = va_arg(args, int);
    Y->n_cols = va_arg(args, int);
    Y->val = va_arg(args, void **);

    return 0;
}

int sptrsm_dispatch(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                    DenseMatrix *Y,
                    std::function<int(KernelContext, SparseMatrix *,
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

int sptrsm_initialize(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                      DenseMatrix *Y) {
    return sptrsm_dispatch(
        context, A, X, Y,
        [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y) {
            return SPTRSM::sptrsm_initialize_cpu(context, A, X, Y);
        },
        "sptrsm_initialize");
}

int sptrsm_apply(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                 DenseMatrix *Y) {
    return sptrsm_dispatch(
        context, A, X, Y,
        [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y) {
            return SPTRSM::sptrsm_apply_cpu(context, A, X, Y);
        },
        "sptrsm_apply");
}

int sptrsm_finalize(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                    DenseMatrix *Y) {
    return sptrsm_dispatch(
        context, A, X, Y,
        [](auto context, SparseMatrix *A, DenseMatrix *X, DenseMatrix *Y) {
            return SPTRSM::sptrsm_finalize_cpu(context, A, X, Y);
        },
        "sptrsm_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSM_HPP
