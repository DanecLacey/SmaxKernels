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

int spgemv_register_B(SparseVector *x, va_list args) {
    x->n_rows = va_arg(args, int);
    x->nnz = va_arg(args, int);
    x->idx = va_arg(args, void **);
    x->val = va_arg(args, void **);

    return 0;
}

int spgemv_register_C(SparseVectorRef *y, va_list args) {
    y->n_rows = va_arg(args, int *);
    y->nnz = va_arg(args, int *);
    y->idx = va_arg(args, void **);
    y->val = va_arg(args, void **);

    return 0;
}

int spgemv_dispatch(
    KernelContext context, SPGEMV::Args *args, SPGEMV::Flags *flags,
    std::function<int(KernelContext, SPGEMV::Args *, SPGEMV::Flags *)> cpu_func,
    const char *label) {
    switch (context.platform_type) {
    case SMAX::CPU:
        CHECK_ERROR(cpu_func(context, args, flags), label);
        break;
    default:
        std::cerr << "Error: Platform not supported\n";
        return 1;
    }
    return 0;
}

int spgemv_initialize(KernelContext context, SPGEMV::Args *args,
                      SPGEMV::Flags *flags) {
    return spgemv_dispatch(
        context, args, flags,
        [](KernelContext context, SPGEMV::Args *args, SPGEMV::Flags *flags) {
            return SPGEMV::spgemv_initialize_cpu(context, args, flags);
        },
        "spgemv_initialize");
}

int spgemv_apply(KernelContext context, SPGEMV::Args *args,
                 SPGEMV::Flags *flags) {
    return spgemv_dispatch(
        context, args, flags,
        [](KernelContext context, SPGEMV::Args *args, SPGEMV::Flags *flags) {
            return SPGEMV::spgemv_apply_cpu(context, args, flags);
        },
        "spgemv_apply");
}

int spgemv_finalize(KernelContext context, SPGEMV::Args *args,
                    SPGEMV::Flags *flags) {
    return spgemv_dispatch(
        context, args, flags,
        [](KernelContext context, SPGEMV::Args *args, SPGEMV::Flags *flags) {
            return SPGEMV::spgemv_finalize_cpu(context, args, flags);
        },
        "spgemv_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMV_HPP
