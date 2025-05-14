#pragma once

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX::KERNELS::SPMV {

int register_A(SparseMatrix *A, va_list args) {
    A->n_rows = va_arg(args, int);
    A->n_cols = va_arg(args, int);
    A->nnz = va_arg(args, int);
    A->col = va_arg(args, void **);
    A->row_ptr = va_arg(args, void **);
    A->val = va_arg(args, void **);

    return 0;
}

int register_B(DenseMatrix *x, va_list args) {
    x->n_rows = va_arg(args, int);
    x->val = va_arg(args, void **);

    return 0;
}

int register_C(DenseMatrix *y, va_list args) {
    y->n_rows = va_arg(args, int);
    y->val = va_arg(args, void **);

    return 0;
}

struct Args {

    SparseMatrix *A;
    DenseMatrix *x;
    DenseMatrix *y;
    UtilitiesContainer *uc;

    Args(UtilitiesContainer *_uc) {
        A = new SparseMatrix();
        x = new DenseMatrix();
        y = new DenseMatrix();
        uc = _uc;
    }

    // Destructor
    ~Args() {
        delete A;
        delete x;
        delete y;
    }

    // Disable copying to prevent double deletion
    Args(const Args &) = delete;
    Args &operator=(const Args &) = delete;
};

struct Flags {};

class SpMVErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpMV");
    }
};

} // namespace SMAX::KERNELS::SPMV
