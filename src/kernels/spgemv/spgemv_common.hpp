#ifndef SMAX_SPGEMV_COMMON_HPP
#define SMAX_SPGEMV_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMV {

struct Args {

    SparseMatrix *A;
    SparseVector *x;
    SparseVectorRef *y;

    Args() {
        A = new SparseMatrix();
        x = new SparseVector();
        y = new SparseVectorRef();
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

class SpGEMVErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpGEMV");
    }

    static void not_implemented() {
        KernelErrorHandler::not_implemented("SpGEMV");
    }
};

} // namespace SPGEMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMV_COMMON_HPP
