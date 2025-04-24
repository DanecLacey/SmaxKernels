#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

int main(void)
{
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 5;
    u_int16_t *A_col = new u_int16_t[A_nnz]{0, 1, 1, 0, 2};
    u_int16_t *A_row_ptr = new u_int16_t[A_n_rows + 1]{0, 2, 3, 5};
    float *A_val = new float[A_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    int B_n_rows = 3;
    int B_n_cols = 3;
    int B_nnz = 5;
    u_int16_t *B_col = new u_int16_t[A_nnz]{0, 2, 0, 1, 0};
    u_int16_t *B_row_ptr = new u_int16_t[A_n_rows + 1]{0, 2, 4, 5};
    float *B_val = new float[A_nnz]{1.1, 1.3, 2.1, 2.2, 3.1};

    int C_n_rows = 0;
    int C_n_cols = 0;
    int C_nnz = 0;
    u_int16_t *C_col = nullptr;
    u_int16_t *C_row_ptr = nullptr;
    float *C_val = nullptr;

    SMAX::Interface *spgemm_kernel = new SMAX::Interface(SMAX::SPGEMM, SMAX::CPU, SMAX::UINT16, SMAX::FLOAT32);
    spgemm_kernel->register_A(&A_n_rows, &A_n_cols, &A_nnz, &A_col, &A_row_ptr, &A_val);
    spgemm_kernel->register_B(&B_n_rows, &B_n_cols, &B_nnz, &B_col, &B_row_ptr, &B_val);
    spgemm_kernel->register_C(&C_n_rows, &C_n_cols, &C_nnz, &C_col, &C_row_ptr, &C_val);

    spgemm_kernel->run();

    spgemm_kernel->print_timers();

    print_matrix<u_int16_t, float>(C_n_rows, C_n_cols, C_nnz, C_col, C_row_ptr, C_val);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] B_col;
    delete[] B_row_ptr;
    delete[] B_val;
    delete[] C_col;
    delete[] C_row_ptr;
    delete[] C_val;
    delete spgemm_kernel;

    return 0;
}