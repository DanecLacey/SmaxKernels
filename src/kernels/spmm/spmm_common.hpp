#pragma once

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX::KERNELS::SPMM {

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

struct Flags {
    bool vec_row_major = false;
};

class SpMMErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, ULL j, ULL A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpMM");
    }

    template <typename IT, typename VT>
    static void print_crs_elem(VT val, IT col, VT X, IT j, IT X_idx) {
        // clang-format off
        std::cout << "A_val[" << j << "] = " << val << std::endl;
        std::cout << "A_col[" << j << "] = " << col << std::endl;
        std::cout << "(A_n_rows * vec_idx) + A_col[" << j << "] = " << X_idx << std::endl;
        std::cout << "X[(A_n_rows * vec_idx) + A_col[" << j << "] = " << X << std::endl;
        // clang-format on
    }
};

} // namespace SMAX::KERNELS::SPMM
