/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse triangular solve with a lower triangular matrix (SpTSV).
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

int main(void) {
    // Initialize operands
    CRSMatrix<int, double> *L = new CRSMatrix<int, double>;
    L->n_rows = 3;
    L->n_cols = 3;
    L->nnz = 4;
    L->col = new int[L->nnz]{0, 1, 0, 2};
    L->row_ptr = new int[L->n_rows + 1]{0, 1, 2, 4};
    L->val = new double[L->nnz]{1.1, 2.2, 3.1, 3.3};

    DenseMatrix<double> *x = new DenseMatrix<double>(L->n_cols, 1, 1.0);

    // Initialize RHS
    DenseMatrix<double> *b = new DenseMatrix<double>(L->n_cols, 1, 2.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("solve_Lx=b", SMAX::KernelType::SPTRSV);

    // Register operands to this kernel tag
    // L is assumed to be in CRS format
    smax->kernel("solve_Lx=b")
        ->register_A(L->n_rows, L->n_cols, L->nnz, L->col, L->row_ptr, L->val);
    // x and b are dense vectors
    smax->kernel("solve_Lx=b")->register_B(L->n_rows, x->val);
    smax->kernel("solve_Lx=b")->register_C(L->n_cols, b->val);

    // Execute all phases of this kernel
    smax->kernel("solve_Lx=b")->run();

    smax->utils->print_timers();

    x->print();

    delete L;
    delete x;
    delete b;
    delete smax;

    return 0;
}