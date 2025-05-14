#pragma once

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX::KERNELS::SPGEMM {

int register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, int);
    A->n_cols = va_arg(args, int);
    A->nnz = va_arg(args, int);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int register_B(SparseMatrix *B, va_list args) {
    B->n_rows = va_arg(args, int);
    B->n_cols = va_arg(args, int);
    B->nnz = va_arg(args, int);
    B->col = va_arg(args, void **);
    B->row_ptr = va_arg(args, void **);
    B->val = va_arg(args, void **);

    return 0;
}

int register_C(SparseMatrixRef *C, va_list args) {
    C->n_rows = va_arg(args, int *);
    C->n_cols = va_arg(args, int *);
    C->nnz = va_arg(args, int *);
    C->col = va_arg(args, void **);
    C->row_ptr = va_arg(args, void **);
    C->val = va_arg(args, void **);

    return 0;
}

struct Args {

    SparseMatrix *A;
    SparseMatrix *B;
    SparseMatrixRef *C;
    UtilitiesContainer *uc;

    Args(UtilitiesContainer *_uc) {
        A = new SparseMatrix();
        B = new SparseMatrix();
        C = new SparseMatrixRef();
        uc = _uc;
    }

    // Destructor
    ~Args() {
        delete A;
        delete B;
        delete C;
    }

    // Disable copying to prevent double deletion
    Args(const Args &) = delete;
    Args &operator=(const Args &) = delete;
};

struct Flags {};

class SpGEMMErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpGEMM");
    }

    static void multithreaded_issue() {
        KernelErrorHandler::issue("Multithreaded problems", "SpGEMM");
    }
};

} // namespace SMAX::KERNELS::SPGEMM
