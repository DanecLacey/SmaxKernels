#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

// Does not work! SMAX needs locations in memory, not literals
// #define N_DENSE_COLS 64

int main(void)
{
    // Initialize operands
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 5;
    int *A_col = new int[A_nnz]{0, 1, 1, 0, 2};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 2, 3, 5};
    double *A_val = new double[A_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    int n_dense_cols = 2;
    double *X = new double[A_n_cols * n_dense_cols];
    for (int i = 0; i < A_n_cols * n_dense_cols; ++i)
    {
        X[i] = 1.0;
    }

    // Initialize result
    double *Y = new double[A_n_cols * n_dense_cols];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel tag, platform, and metadata
    smax->register_kernel("my_spmv", SMAX::SPMV, SMAX::CPU);

    // Register operands to this kernel tag
    // A is assumed to be in CRS format
    smax->kernels["my_spmv"]->register_A(
        &A_n_rows, &A_n_cols, &A_nnz, &A_col, &A_row_ptr, &A_val);
    // X and Y are dense matrices
    smax->kernels["my_spmv"]->register_B(&A_n_cols, &n_dense_cols, &X);
    smax->kernels["my_spmv"]->register_C(&A_n_cols, &n_dense_cols, &Y);

    // Execute all phases of this kernel
    smax->kernels["my_spmv"]->run();

    smax->print_timers();

    print_vector<double>(Y, A_n_cols * n_dense_cols);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] X;
    delete[] Y;
    delete smax;

    return 0;
}