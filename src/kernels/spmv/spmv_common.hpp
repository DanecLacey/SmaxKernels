#ifndef SMAX_SPMV_COMMON_HPP
#define SMAX_SPMV_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMV {

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

} // namespace SPMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMV_COMMON_HPP
