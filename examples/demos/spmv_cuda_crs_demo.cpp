/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-vector multiplication (SpMV) using the CUDA platform.
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

int main(void) {
    // Initialize operands
    CRSMatrix<int, double> *A = new CRSMatrix<int, double>;
    A->n_rows = 3;
    A->n_cols = 3;
    A->nnz = 5;
    A->col = new int[A->nnz]{0, 1, 1, 0, 2};
    A->row_ptr = new int[A->n_rows + 1]{0, 2, 3, 5};
    A->val = new double[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    DenseMatrix<double> *x = new DenseMatrix<double>(A->n_cols, 1, 1.0);

    // Initialize result
    DenseMatrix<double> *y = new DenseMatrix<double>(A->n_rows, 1, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("my_spmv", SMAX::KernelType::SPMV,
                          SMAX::PlatformType::CUDA);

    // Register operands to this kernel tag
    // A is assumed to be in CRS format
    smax->kernel("my_spmv")->register_A(A->n_rows, A->n_cols, A->nnz, A->col,
                                        A->row_ptr, A->val);
    // x and y are dense matrices
    smax->kernel("my_spmv")->register_B(A->n_cols, x->val);
    smax->kernel("my_spmv")->register_C(A->n_rows, y->val);

    // Execute all phases of this kernel
    smax->kernel("my_spmv")->run();

    smax->utils->print_timers();

    y->print();

    delete A;
    delete x;
    delete y;
    delete smax;

    return 0;
}