/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse triangular solve with a lower triangular matrix and multiple RHS
 * vectors (SpTSM).
 */
#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

#define N_VECTORS 4

int main(void) {
    // Initialize operands
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 4;
    int *A_col = new int[A_nnz]{0, 1, 0, 2};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 1, 2, 4};
    float *A_val = new float[A_nnz]{1.1, 2.2, 3.1, 3.3};

    float *X = new float[A_n_cols * N_VECTORS];
    for (int i = 0; i < A_n_cols * N_VECTORS; ++i) {
        X[i] = 1.0;
    }

    // Initialize RHS
    float *B = new float[A_n_cols * N_VECTORS];
    for (int i = 0; i < A_n_cols * N_VECTORS; ++i) {
        B[i] = 2.0;
    }

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("solve_LX=B", SMAX::KernelType::SPTRSM,
                          SMAX::PlatformType::CPU, SMAX::IntType::UINT32,
                          SMAX::FloatType::FLOAT32);

    // Register operands to this kernel tag
    // A is assumed to be in CRS format
    smax->kernel("solve_LX=B")
        ->register_A(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);
    // X and B are dense vectors
    smax->kernel("solve_LX=B")->register_B(A_n_rows, N_VECTORS, X);
    smax->kernel("solve_LX=B")->register_C(A_n_cols, N_VECTORS, B);

    // Execute all phases of this kernel
    smax->kernel("solve_LX=B")->run();

    smax->utils->print_timers();

    print_vector<float>(X, A_n_cols * N_VECTORS);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] X;
    delete[] B;
    delete smax;

    return 0;
}