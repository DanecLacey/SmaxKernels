#pragma once

#include "../../common.hpp"
#include "../kernels_common.hpp"

namespace SMAX::KERNELS::BSPMV {

struct Args {

    BlockedSparseMatrix *A;
    DenseMatrix *x;
    DenseMatrix *y;
    BlockedSparseMatrix *d_A; // Device copy
    DenseMatrix *d_x;  // Device copy
    DenseMatrix *d_y;  // Device copy
    UtilitiesContainer *uc;
    // ULL block_height;
    // ULL block_width;

    Args(UtilitiesContainer *_uc) {
        A = new BlockedSparseMatrix();
        x = new DenseMatrix();
        y = new DenseMatrix();
        d_A = new BlockedSparseMatrix();
        d_x = new DenseMatrix();
        d_y = new DenseMatrix();
        uc = _uc;
    }

    // Destructor
    ~Args() {
        delete A;
        delete x;
        delete y;
        delete d_A;
        delete d_x;
        delete d_y;
    }

    // Disable copying to prevent double deletion
    Args(const Args &) = delete;
    Args &operator=(const Args &) = delete;
};

struct Flags {
    BCRSKernelType kernel_type = BCRSKernelType::naive_thread_per_row;
    bool is_mat_scs = false;
    bool is_block_column_major = false;
};

class BSpMVErrorHandler : public KernelErrorHandler {
  public:
    template <typename IT>
    static void col_oob(IT col_value, ULL j, ULL A_n_cols) {
        KernelErrorHandler::col_oob<IT>(col_value, j, A_n_cols, "BSpMV");
    }

    template <typename IT, typename VT>
    static void print_bcrs_elem(VT val, IT col, VT x, IT j, IT ix, IT jx) {
        std::cout << "A_val[" << j << "," << ix << "," << "] = " << val << std::endl;
        std::cout << "A_col[" << j << "," << ix << "," << "] = " << col << std::endl;
        std::cout << "x[A_col[" << j << "," << ix << "," << "] = " << x << std::endl;
    }
};

} // namespace SMAX::KERNELS::SPMV