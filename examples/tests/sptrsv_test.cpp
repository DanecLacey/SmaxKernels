/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse triangular solve with a lower triangular matrix (SpTSV).
 */
#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

int main(void) {
    // Initialize operands
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 4;
    int *A_col = new int[A_nnz]{0, 1, 0, 2};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 1, 2, 4};
    double *A_val = new double[A_nnz]{1.1, 2.2, 3.1, 3.3};

    double *x = new double[A_n_cols];
    for (int i = 0; i < A_n_cols; ++i) {
        x[i] = 1.0;
    }

    // Initialize RHS
    double *b = new double[A_n_cols];
    for (int i = 0; i < A_n_cols; ++i) {
        b[i] = 2.0;
    }

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("solve_Lx=b", SMAX::KernelType::SPTRSV);

    // Register operands to this kernel tag
    // A is assumed to be in CRS format
    smax->kernel("solve_Lx=b")
        ->register_A(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);
    // x and b are dense vectors
    smax->kernel("solve_Lx=b")->register_B(A_n_rows, x);
    smax->kernel("solve_Lx=b")->register_C(A_n_cols, b);

    // Execute all phases of this kernel
    smax->kernel("solve_Lx=b")->run();

    smax->utils->print_timers();

    print_vector<double>(x, A_n_cols);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] x;
    delete[] b;
    delete smax;

    return 0;
}