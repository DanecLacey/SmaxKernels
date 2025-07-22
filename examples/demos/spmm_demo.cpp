/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-multiple vector multiplication (SpMM).
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

#define N_VECTORS 4

int main(void) {
    // Initialize operands
    CRSMatrix<int, double> *A = new CRSMatrix<int, double>;
    A->n_rows = 3;
    A->n_cols = 3;
    A->nnz = 5;
    A->col = new int[A->nnz]{0, 1, 1, 0, 2};
    A->row_ptr = new int[A->n_rows + 1]{0, 2, 3, 5};
    A->val = new double[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    DenseMatrix<double> *X = new DenseMatrix<double>(A->n_cols, N_VECTORS, 1.0);
    bool rowwise = true;

    // Initialize result
    DenseMatrix<double> *Y = new DenseMatrix<double>(A->n_rows, N_VECTORS, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("useful_spmm", SMAX::KernelType::SPMM);

    // Register operands to this kernel tag
    // A is assumed to be in CRS format
    smax->kernel("useful_spmm")
        ->register_A(A->n_rows, A->n_cols, A->nnz, A->col, A->row_ptr, A->val);
    // X and Y are dense matrices
    smax->kernel("useful_spmm")->register_B(A->n_cols, N_VECTORS, X->val);
    smax->kernel("useful_spmm")->register_C(A->n_rows, N_VECTORS, Y->val);

    // Optionally, tell SMAX how to access dense block vector
    smax->kernel("useful_spmm")->set_vec_row_major(rowwise);

    // Execute all phases of this kernel
    smax->kernel("useful_spmm")->run();

    smax->utils->print_timers();

    Y->print();

    delete A;
    delete X;
    delete Y;
    delete smax;

    return 0;
}