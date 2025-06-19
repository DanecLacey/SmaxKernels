#pragma once

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX::KERNELS::SPGEMM {

struct Args {

    SparseMatrix *A;
    SparseMatrix *B;
    SparseMatrixRef *C;
    UtilitiesContainer *uc;

    Args(UtilitiesContainer *_uc) {
        A = new SparseMatrix();
        B = new SparseMatrix();
        C = new SparseMatrixRef();
        uc = _uc;
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
    static void col_oob(IT col_value, ULL j, ULL A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "SpGEMM");
    }

    static void multithreaded_issue() {
        KernelErrorHandler::issue("Multithreaded problems", "SpGEMM");
    }
};

} // namespace SMAX::KERNELS::SPGEMM
