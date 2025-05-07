#ifndef SMAX_SPMM_COMMON_HPP
#define SMAX_SPMM_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMM {

struct Args {

    SparseMatrix *A;
    DenseMatrix *X;
    DenseMatrix *Y;

    Args() {
        A = new SparseMatrix();
        X = new DenseMatrix();
        Y = new DenseMatrix();
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

} // namespace SPMM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMM_COMMON_HPP
