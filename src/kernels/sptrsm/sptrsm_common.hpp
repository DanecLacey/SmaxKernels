#pragma once

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX::KERNELS::SPTRSM {

struct Args {

    SparseMatrix *A;
    DenseMatrix *D;
    DenseMatrix *X;
    DenseMatrix *Y;
    UtilitiesContainer *uc;

    Args(UtilitiesContainer *_uc) {
        A = new SparseMatrix();
        D = new DenseMatrix();
        X = new DenseMatrix();
        Y = new DenseMatrix();
        uc = _uc;
    }

    // Destructor
    ~Args() {
        delete A;
        delete D;
        delete X;
        delete Y;
    }

    // Disable copying to prevent double deletion
    Args(const Args &) = delete;
    Args &operator=(const Args &) = delete;
};

struct Flags {
    bool mat_permuted = false;
    bool mat_upper_triang = false;
    bool mat_lower_triang = false;
    bool diag_collected = false;
};

class SpTRSMErrorHandler : public KernelErrorHandler {
  public:
    static void zero_diag(int row_idx) {
        std::ostringstream oss;
        oss << "Zero detected on diagonal at row index" << row_idx;
        kernel_fatal("[SpTRSM] " + oss.str());
    }

    static void no_diag(int row_idx) {
        std::ostringstream oss;
        oss << "No diagonal to extract at row index" << row_idx;
        kernel_fatal("[SpTRSM] " + oss.str());
    }

    template <typename IT, typename VT>
    static void super_diag(int row_idx, IT col, VT val) {
        KernelErrorHandler::super_diag<IT>(row_idx, col, val, "SpTRSM");
    }

    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpTRSM");
    }

    static void not_implemented() {
        KernelErrorHandler::not_implemented("SpTRSM");
    }
};

} // namespace SMAX::KERNELS::SPTRSM
