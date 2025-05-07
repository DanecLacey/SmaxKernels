#ifndef SMAX_SPGEMM_COMMON_HPP
#define SMAX_SPGEMM_COMMON_HPP

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {

struct Args {

    SparseMatrix *A;
    SparseMatrix *B;
    SparseMatrixRef *C;

    Args() {
        A = new SparseMatrix();
        B = new SparseMatrix();
        C = new SparseMatrixRef();
    }

    // Destructor
    ~Args() {
        delete A;
        delete B;
        delete C;
    }

    // Disable copying to prevent double deletion
    Args(const Args &) = delete;
    Args &operator=(const Args &) = delete;
};

struct Flags {};

class SpGEMMErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, int j, int A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpGEMM");
    }

    static void multithreaded_issue() {
        KernelErrorHandler::issue("Multithreaded problems", "SpGEMM");
    }
};

} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMM_COMMON_HPP
