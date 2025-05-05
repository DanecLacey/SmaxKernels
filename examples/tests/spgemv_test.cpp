/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-sparse vector multiplication (SpGEMV).
 */
#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

int main(void) {
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 5;
    int *A_col = new int[A_nnz]{0, 1, 1, 0, 2};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 2, 3, 5};
    double *A_val = new double[A_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    int X_n_rows = A_n_rows;
    int X_nnz = 2;
    int *X_idx = new int[X_nnz]{0, 2};
    double *X_val = new double[X_nnz]{1.0, 3.0};

    int Y_n_rows = 0;
    int Y_nnz = 0;
    int *Y_idx = nullptr;
    double *Y_val = nullptr;

    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("fast_spgemv", SMAX::SPGEMV, SMAX::CPU);

    smax->kernels["fast_spgemv"]->register_A(A_n_rows, A_n_cols, A_nnz, &A_col,
                                             &A_row_ptr, &A_val);
    smax->kernels["fast_spgemv"]->register_B(X_n_rows, X_nnz, &X_idx, &X_val);
    smax->kernels["fast_spgemv"]->register_C(&Y_n_rows, &Y_nnz, &Y_idx, &Y_val);

    smax->kernels["fast_spgemv"]->run();

    smax->print_timers();

    print_sparse_vector<int, double>(Y_n_rows, Y_nnz, Y_idx, Y_val);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] X_idx;
    delete[] X_val;
    delete[] Y_idx;
    delete[] Y_val;
    delete smax;

    return 0;
}