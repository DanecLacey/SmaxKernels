/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-matrix multiplication (SpGEMM) with the two input matrices
 * being the same (A == B).
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

using ULL = unsigned long long int;

int main(void) {
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 5;
    u_int16_t *A_col = new u_int16_t[A_nnz]{0, 1, 1, 0, 2};
    u_int16_t *A_row_ptr = new u_int16_t[A_n_rows + 1]{0, 2, 3, 5};
    float *A_val = new float[A_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    // NOTE: We require metadata of C is of type "unsigned long long int"
    ULL C_n_rows = 0;
    ULL C_n_cols = 0;
    ULL C_nnz = 0;
    u_int16_t *C_col = nullptr;
    u_int16_t *C_row_ptr = nullptr;
    float *C_val = nullptr;

    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("my_spgemm_AA", SMAX::KernelType::SPGEMM,
                          SMAX::PlatformType::CPU, SMAX::IntType::UINT16,
                          SMAX::FloatType::FLOAT32);

    smax->kernel("my_spgemm_AA")
        ->register_A(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);
    smax->kernel("my_spgemm_AA")
        ->register_B(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);
    smax->kernel("my_spgemm_AA")
        ->register_C(&C_n_rows, &C_n_cols, &C_nnz, &C_col, &C_row_ptr, &C_val);

    smax->kernel("my_spgemm_AA")->run();

    smax->utils->print_timers();

    print_matrix<u_int16_t, float>(C_n_rows, C_n_cols, C_nnz, C_col, C_row_ptr,
                                   C_val);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] C_col;
    delete[] C_row_ptr;
    delete[] C_val;
    delete smax;

    return 0;
}