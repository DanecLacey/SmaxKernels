#pragma once

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX::KERNELS::SPMV {

struct Args {

    SparseMatrix *A;
    DenseMatrix *x;
    DenseMatrix *y;
    SparseMatrix *d_A; // Device copy
    DenseMatrix *d_x;  // Device copy
    DenseMatrix *d_y;  // Device copy
    UtilitiesContainer *uc;

    Args(UtilitiesContainer *_uc) {
        A = new SparseMatrix();
        x = new DenseMatrix();
        y = new DenseMatrix();
        d_A = new SparseMatrix();
        d_x = new DenseMatrix();
        d_y = new DenseMatrix();
        uc = _uc;
    }

    // Destructor
    ~Args() {
        delete A;
        delete x;
        delete y;
        delete d_A;
        delete d_x;
        delete d_y;
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

    template <typename IT, typename VT>
    static void print_crs_elem(VT val, IT col, VT x, IT j) {
        std::cout << "A_val[" << j << "] = " << val << std::endl;
        std::cout << "A_col[" << j << "] = " << col << std::endl;
        std::cout << "x[A_col[" << j << "] = " << x << std::endl;
    }
};

} // namespace SMAX::KERNELS::SPMV
