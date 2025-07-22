/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse triangular solve with a lower triangular matrix and multiple RHS
 * vectors (SpTSM).
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

#define N_VECTORS 4

int main(void) {
    // Initialize operands
    CRSMatrix<int, float> *L = new CRSMatrix<int, float>;
    L->n_rows = 3;
    L->n_cols = 3;
    L->nnz = 4;
    L->col = new int[L->nnz]{0, 1, 0, 2};
    L->row_ptr = new int[L->n_rows + 1]{0, 1, 2, 4};
    L->val = new float[L->nnz]{1.1, 2.2, 3.1, 3.3};

    DenseMatrix<float> *X = new DenseMatrix<float>(L->n_cols, N_VECTORS, 1.0);

    // Initialize RHS
    DenseMatrix<float> *B = new DenseMatrix<float>(L->n_rows, N_VECTORS, 2.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("solve_LX=B", SMAX::KernelType::SPTRSM,
                          SMAX::PlatformType::CPU, SMAX::IntType::INT32,
                          SMAX::FloatType::FLOAT32);

    // Register operands to this kernel tag
    // L is assumed to be in CRS format
    smax->kernel("solve_LX=B")
        ->register_A(L->n_rows, L->n_cols, L->nnz, L->col, L->row_ptr, L->val);
    // X and B are dense vectors
    smax->kernel("solve_LX=B")->register_B(L->n_rows, N_VECTORS, X->val);
    smax->kernel("solve_LX=B")->register_C(L->n_cols, N_VECTORS, B->val);

    // Execute all phases of this kernel
    smax->kernel("solve_LX=B")->run();

    smax->utils->print_timers();

    X->print();

    delete L;
    delete X;
    delete B;
    delete smax;

    return 0;
}