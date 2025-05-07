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

int sptrsm_dispatch(
    KernelContext context, SPTRSM::Args *args, SPTRSM::Flags *flags,
    std::function<int(KernelContext, SPTRSM::Args *, SPTRSM::Flags *)> cpu_func,
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

int sptrsm_initialize(KernelContext context, SPTRSM::Args *args,
                      SPTRSM::Flags *flags) {
    return sptrsm_dispatch(
        context, args, flags,
        [](auto context, SPTRSM::Args *args, SPTRSM::Flags *flags) {
            return SPTRSM::sptrsm_initialize_cpu(context, args, flags);
        },
        "sptrsm_initialize");
}

int sptrsm_apply(KernelContext context, SPTRSM::Args *args,
                 SPTRSM::Flags *flags) {
    return sptrsm_dispatch(
        context, args, flags,
        [](auto context, SPTRSM::Args *args, SPTRSM::Flags *flags) {
            return SPTRSM::sptrsm_apply_cpu(context, args, flags);
        },
        "sptrsm_apply");
}

int sptrsm_finalize(KernelContext context, SPTRSM::Args *args,
                    SPTRSM::Flags *flags) {
    return sptrsm_dispatch(
        context, args, flags,
        [](auto context, SPTRSM::Args *args, SPTRSM::Flags *flags) {
            return SPTRSM::sptrsm_finalize_cpu(context, args, flags);
        },
        "sptrsm_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSM_HPP
