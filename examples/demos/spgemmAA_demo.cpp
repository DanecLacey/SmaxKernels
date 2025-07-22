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
    CRSMatrix<u_int16_t, float> *A = new CRSMatrix<u_int16_t, float>;
    A->n_rows = 3;
    A->n_cols = 3;
    A->nnz = 5;
    A->col = new u_int16_t[A->nnz]{0, 1, 1, 0, 2};
    A->row_ptr = new u_int16_t[A->n_rows + 1]{0, 2, 3, 5};
    A->val = new float[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    CRSMatrix<u_int16_t, float> *C = new CRSMatrix<u_int16_t, float>;

    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("my_spgemm_AA", SMAX::KernelType::SPGEMM,
                          SMAX::PlatformType::CPU, SMAX::IntType::UINT16,
                          SMAX::FloatType::FLOAT32);

    smax->kernel("my_spgemm_AA")
        ->register_A(A->n_rows, A->n_cols, A->nnz, A->col, A->row_ptr, A->val);
    smax->kernel("my_spgemm_AA")
        ->register_B(A->n_rows, A->n_cols, A->nnz, A->col, A->row_ptr, A->val);
    smax->kernel("my_spgemm_AA")
        ->register_C(&C->n_rows, &C->n_cols, &C->nnz, &C->col, &C->row_ptr,
                     &C->val);

    smax->kernel("my_spgemm_AA")->run();

    smax->utils->print_timers();

    C->print();

    delete A;
    delete C;
    delete smax;

    return 0;
}