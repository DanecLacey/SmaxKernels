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
    int *lvl_ptr;
    int n_levels;

    Args() {
        A = new SparseMatrix();
        x = new DenseMatrix();
        y = new DenseMatrix();
        lvl_ptr = new int[A->n_rows];
    }

    // Destructor
    ~Args() {
        delete A;
        delete x;
        delete y;
        delete lvl_ptr;
    }

    // Disable copying to prevent double deletion
    Args(const Args &) = delete;
    Args &operator=(const Args &) = delete;
};

struct Flags {
    bool mat_permuted = false;
    bool lvl_ptr_collected = false;
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
};

} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_COMMON_HPP
