#ifndef SMAX_SPMV_HPP
#define SMAX_SPMV_HPP

#include "../common.hpp"
#include "../macros.hpp"
#include "spmv/spmv_cpu.hpp"

#include <cstdarg>
#include <functional>

namespace SMAX {
namespace KERNELS {

int spmv_register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, int);
    A->n_cols = va_arg(args, int);
    A->nnz = va_arg(args, int);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int spmv_register_B(DenseMatrix *x, va_list args) {
    x->n_rows = va_arg(args, int);
    x->val = va_arg(args, void **);

    return 0;
}

int spmv_register_C(DenseMatrix *y, va_list args) {
    y->n_rows = va_arg(args, int);
    y->val = va_arg(args, void **);

    return 0;
}

int spmv_dispatch(KernelContext context, SPMV::Args *args, SPMV::Flags *flags,
                  int A_offset, int x_offset, int y_offset,
                  std::function<int(KernelContext, SPMV::Args *, SPMV::Flags *,
                                    int, int, int)>
                      cpu_func,
                  const char *label) {
    switch (context.platform_type) {
    case CPU:
        CHECK_ERROR(
            cpu_func(context, args, flags, A_offset, x_offset, y_offset),
            label);
        break;
    default:
        std::cerr << "Error: Platform not supported\n";
        return 1;
    }
    return 0;
}

int spmv_initialize(KernelContext context, SPMV::Args *args, SPMV::Flags *flags,
                    int A_offset, int x_offset, int y_offset) {

    return spmv_dispatch(
        context, args, flags, A_offset, x_offset, y_offset,
        [](KernelContext context, SPMV::Args *args, SPMV::Flags *flags,
           int A_offset, int x_offset, int y_offset) {
            return SPMV::spmv_initialize_cpu(context, args, flags, A_offset,
                                             x_offset, y_offset);
        },
        "spmv_initialize");
}

int spmv_apply(KernelContext context, SPMV::Args *args, SPMV::Flags *flags,
               int A_offset, int x_offset, int y_offset) {

    return spmv_dispatch(
        context, args, flags, A_offset, x_offset, y_offset,
        [](KernelContext context, SPMV::Args *args, SPMV::Flags *flags,
           int A_offset, int x_offset, int y_offset) {
            return SPMV::spmv_apply_cpu(context, args, flags, A_offset,
                                        x_offset, y_offset);
        },
        "spmv_apply");
}

int spmv_finalize(KernelContext context, SPMV::Args *args, SPMV::Flags *flags,
                  int A_offset, int x_offset, int y_offset) {

    return spmv_dispatch(
        context, args, flags, A_offset, x_offset, y_offset,
        [](KernelContext context, SPMV::Args *args, SPMV::Flags *flags,
           int A_offset, int x_offset, int y_offset) {
            return SPMV::spmv_finalize_cpu(context, args, flags, A_offset,
                                           x_offset, y_offset);
        },
        "spmv_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMV_HPP
