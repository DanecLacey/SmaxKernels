#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

// Does not work! SMAX needs locations in memory, not literals
// #define N_DENSE_COLS 64

int main(void)
{
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
    double *Y = new double[A_n_cols * n_dense_cols];

    SMAX::Interface *spmv_kernel = new SMAX::Interface(SMAX::SPMV, SMAX::CPU);
    spmv_kernel->register_A(&A_n_rows, &A_n_cols, &A_nnz, &A_col, &A_row_ptr, &A_val);
    spmv_kernel->register_B(&A_n_cols, &n_dense_cols, &X);
    spmv_kernel->register_C(&A_n_cols, &n_dense_cols, &Y);

    spmv_kernel->run();

    spmv_kernel->print_timers();

    print_vector<double>(Y, A_n_cols * n_dense_cols);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] X;
    delete[] Y;
    delete spmv_kernel;

    return 0;
}