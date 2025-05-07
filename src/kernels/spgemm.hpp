#ifndef SMAX_SPGEMM_HPP
#define SMAX_SPGEMM_HPP

#include "../common.hpp"
#include "../macros.hpp"
#include "spgemm/spgemm_cpu.hpp"

#include <cstdarg>
#include <functional>

namespace SMAX {
namespace KERNELS {

int spgemm_register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, int);
    A->n_cols = va_arg(args, int);
    A->nnz = va_arg(args, int);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int spgemm_register_B(SparseMatrix *B, va_list args) {
    B->n_rows = va_arg(args, int);
    B->n_cols = va_arg(args, int);
    B->nnz = va_arg(args, int);
    B->col = va_arg(args, void **);
    B->row_ptr = va_arg(args, void **);
    B->val = va_arg(args, void **);

    return 0;
}

int spgemm_register_C(SparseMatrixRef *C, va_list args) {
    C->n_rows = va_arg(args, int *);
    C->n_cols = va_arg(args, int *);
    C->nnz = va_arg(args, int *);
    C->col = va_arg(args, void **);
    C->row_ptr = va_arg(args, void **);
    C->val = va_arg(args, void **);

    return 0;
}

int spgemm_dispatch(
    KernelContext context, SPGEMM::Args *args, SPGEMM::Flags *flags,
    std::function<int(KernelContext, SPGEMM::Args *, SPGEMM::Flags *)> cpu_func,
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

int spgemm_initialize(KernelContext context, SPGEMM::Args *args,
                      SPGEMM::Flags *flags) {
    return spgemm_dispatch(
        context, args, flags,
        [](auto context, SPGEMM::Args *args, SPGEMM::Flags *flags) {
            return SPGEMM::spgemm_initialize_cpu(context, args, flags);
        },
        "spgemm_initialize");
}

int spgemm_apply(KernelContext context, SPGEMM::Args *args,
                 SPGEMM::Flags *flags) {
    return spgemm_dispatch(
        context, args, flags,
        [](auto context, SPGEMM::Args *args, SPGEMM::Flags *flags) {
            return SPGEMM::spgemm_apply_cpu(context, args, flags);
        },
        "spgemm_apply");
}

int spgemm_finalize(KernelContext context, SPGEMM::Args *args,
                    SPGEMM::Flags *flags) {
    return spgemm_dispatch(
        context, args, flags,
        [](auto context, SPGEMM::Args *args, SPGEMM::Flags *flags) {
            return SPGEMM::spgemm_finalize_cpu(context, args, flags);
        },
        "spgemm_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMM_HPP
