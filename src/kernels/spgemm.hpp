#ifndef SPGEMM_HPP
#define SPGEMM_HPP

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

// This must be treated differently, and we assume that n_rows, n_cols, and nnz
// are in-fact locations in memory and not literals
int spgemm_register_C(SparseMatrixRef *C_ref, va_list args) {
    C_ref->n_rows = va_arg(args, int *);
    C_ref->n_cols = va_arg(args, int *);
    C_ref->nnz = va_arg(args, int *);
    C_ref->col = va_arg(args, void **);
    C_ref->row_ptr = va_arg(args, void **);
    C_ref->val = va_arg(args, void **);

    return 0;
}

int spgemm_dispatch(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                    SparseMatrixRef *C_ref,
                    std::function<int(KernelContext, SparseMatrix *,
                                      SparseMatrix *, SparseMatrixRef *)>
                        cpu_func,
                    const char *label) {
    switch (context.platform_type) {
    case SMAX::CPU:
        CHECK_ERROR(cpu_func(context, A, B, C_ref), label);
        break;
    default:
        std::cerr << "Error: Platform not supported\n";
        return 1;
    }
    return 0;
}

int spgemm_initialize(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                      SparseMatrixRef *C_ref) {
    return spgemm_dispatch(
        context, A, B, C_ref,
        [](auto context, SparseMatrix *A, SparseMatrix *B,
           SparseMatrixRef *C_ref) {
            return SPGEMM::spgemm_initialize_cpu(context, A, B, C_ref);
        },
        "spgemm_initialize");
}

int spgemm_apply(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                 SparseMatrixRef *C_ref) {
    return spgemm_dispatch(
        context, A, B, C_ref,
        [](auto context, SparseMatrix *A, SparseMatrix *B,
           SparseMatrixRef *C_ref) {
            return SPGEMM::spgemm_apply_cpu(context, A, B, C_ref);
        },
        "spgemm_apply");
}

int spgemm_finalize(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                    SparseMatrixRef *C_ref) {
    return spgemm_dispatch(
        context, A, B, C_ref,
        [](auto context, SparseMatrix *A, SparseMatrix *B,
           SparseMatrixRef *C_ref) {
            return SPGEMM::spgemm_finalize_cpu(context, A, B, C_ref);
        },
        "spgemm_finalize");
}

} // namespace KERNELS
} // namespace SMAX

#endif // SPGEMM_HPP
