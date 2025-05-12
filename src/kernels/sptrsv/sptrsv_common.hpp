#ifndef SMAX_SPTRSV_COMMON_HPP
#define SMAX_SPTRSV_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSV {

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

struct Flags {
    bool mat_permuted = false;
    bool mat_upper_triang = false;
    bool mat_lower_triang = false;
    bool diag_collected = false;
};

class SpTRSVErrorHandler : public KernelErrorHandler {
  public:
    static void zero_diag() {
        const std::string message = "Zero detected on diagonal.";
        kernel_fatal("[SpTRSV] " + message);
    }

    static void super_diag() {
        const std::string message = "Nonzero above diagonal detected.";
        kernel_fatal("[SpTRSV] " + message);
    }

    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpTRSV");
    }

    static void levels_issue() {
        KernelErrorHandler::issue("Levels aren't detected", "SpTRSV");
    }

    static void not_validated() {
        KernelErrorHandler::issue("Results not yet validated", "SpTRSV");
    }
};

} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_COMMON_HPP
