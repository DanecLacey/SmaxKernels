#pragma once

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX::KERNELS::SPMM {

int register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, int);
    A->n_cols = va_arg(args, int);
    A->nnz = va_arg(args, int);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int register_B(DenseMatrix *X, va_list args) {
    X->n_rows = va_arg(args, int);
    X->n_cols = va_arg(args, int);
    X->val = va_arg(args, void **);

    return 0;
}

int register_C(DenseMatrix *Y, va_list args) {
    Y->n_rows = va_arg(args, int);
    Y->n_cols = va_arg(args, int);
    Y->val = va_arg(args, void **);

    return 0;
}

struct Args {

    SparseMatrix *A;
    DenseMatrix *X;
    DenseMatrix *Y;
    UtilitiesContainer *uc;

    Args(UtilitiesContainer *_uc) {
        A = new SparseMatrix();
        X = new DenseMatrix();
        Y = new DenseMatrix();
        uc = _uc;
    }

    // Destructor
    ~Args() {
        delete A;
        delete X;
        delete Y;
    }

    // Disable copying to prevent double deletion
    Args(const Args &) = delete;
    Args &operator=(const Args &) = delete;
};

struct Flags {};

class SpMMErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpMM");
    }
};

} // namespace SMAX::KERNELS::SPMM
