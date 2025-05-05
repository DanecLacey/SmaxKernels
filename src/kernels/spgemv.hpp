#ifndef SMAX_SPGEMV_HPP
#define SMAX_SPGEMV_HPP

#include "../common.hpp"
#include "../macros.hpp"
#include "spgemv/spgemv_cpu.hpp"

#include <cstdarg>
#include <functional>

namespace SMAX {
namespace KERNELS {

int spgemv_register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, int);
    A->n_cols = va_arg(args, int);
    A->nnz = va_arg(args, int);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int spgemv_register_B(SparseVector *spX, va_list args) {
    spX->n_rows = va_arg(args, int);
    spX->nnz = va_arg(args, int);
    spX->idx = va_arg(args, void **);
    spX->val = va_arg(args, void **);

    return 0;
}

int spgemv_register_C(SparseVectorRef *spY_ref, va_list args) {
    spY_ref->n_rows = va_arg(args, int *);
    spY_ref->nnz = va_arg(args, int *);
    spY_ref->idx = va_arg(args, void **);
    spY_ref->val = va_arg(args, void **);

    return 0;
}

int spgemv_dispatch(KernelContext context, SparseMatrix *A, SparseVector *spX,
                    SparseVectorRef *spY_ref,
                    std::function<int(KernelContext, SparseMatrix *,
                                      SparseVector *, SparseVectorRef *)>
                        cpu_func,
                    const char *label) {
    switch (context.platform_type) {
    case SMAX::CPU:
        CHECK_ERROR(cpu_func(context, A, spX, spY_ref), label);
        break;
    default:
        std::cerr << "Error: Platform not supported\n";
        return 1;
    }
    return 0;
}

int spgemv_initialize(KernelContext context, SparseMatrix *A, SparseVector *spX,
                      SparseVectorRef *spY_ref) {
    return spgemv_dispatch(
        context, A, spX, spY_ref,
        [](KernelContext context, SparseMatrix *A, SparseVector *spX,
           SparseVectorRef *spY_ref) {
            return SPGEMV::spgemv_initialize_cpu(context, A, spX, spY_ref);
        },
        "spgemv_initialize");
}

int spgemv_apply(KernelContext context, SparseMatrix *A, SparseVector *spX,
                 SparseVectorRef *spY_ref) {
    return spgemv_dispatch(
        context, A, spX, spY_ref,
        [](KernelContext context, SparseMatrix *A, SparseVector *spX,
           SparseVectorRef *spY_ref) {
            return SPGEMV::spgemv_apply_cpu(context, A, spX, spY_ref);
        },
        "spgemv_apply");
}

int spgemv_finalize(KernelContext context, SparseMatrix *A, SparseVector *spX,
                    SparseVectorRef *spY_ref) {
    return spgemv_dispatch(
        context, A, spX, spY_ref,
        [](KernelContext context, SparseMatrix *A, SparseVector *spX,
           SparseVectorRef *spY_ref) {
            return SPGEMV::spgemv_finalize_cpu(context, A, spX, spY_ref);
        },
        "spgemv_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMV_HPP
