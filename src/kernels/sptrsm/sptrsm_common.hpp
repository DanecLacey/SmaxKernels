#ifndef SMAX_SPTRSM_COMMON_HPP
#define SMAX_SPTRSM_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSM {

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
    bool mat_permuted = false;
    bool mat_upper_triang = false;
    bool mat_lower_triang = false;
    bool diag_collected = false;
};

class SpTRSMErrorHandler : public KernelErrorHandler {
  public:
    static void zero_diag() {
        const std::string message = "Zero detected on diagonal.";
        kernel_fatal("[SpTRSM] " + message);
    }

    static void super_diag() {
        const std::string message = "Nonzero above diagonal detected.";
        kernel_fatal("[SpTRSM] " + message);
    }

    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpTRSM");
    }

    static void not_implemented() {
        KernelErrorHandler::not_implemented("SpTRSM");
    }
};

} // namespace SPTRSM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSM_COMMON_HPP
