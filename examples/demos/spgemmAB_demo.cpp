/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-matrix multiplication (SpGEMM) with two different input
 * matrices (A != B). It also shows how one may use different integer and float
 * data types.
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

int main(void) {
    CRSMatrix<int32_t, float> *A = new CRSMatrix<int32_t, float>;
    A->n_rows = 3;
    A->n_cols = 3;
    A->nnz = 5;
    A->col = new int32_t[A->nnz]{0, 1, 1, 0, 2};
    A->row_ptr = new int32_t[A->n_rows + 1]{0, 2, 3, 5};
    A->val = new float[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    CRSMatrix<int32_t, float> *B = new CRSMatrix<int32_t, float>;
    B->n_rows = 3;
    B->n_cols = 3;
    B->nnz = 5;
    B->col = new int32_t[B->nnz]{0, 2, 0, 1, 0};
    B->row_ptr = new int32_t[B->n_rows + 1]{0, 2, 4, 5};
    B->val = new float[B->nnz]{1.1, 1.3, 2.1, 2.2, 3.1};

    CRSMatrix<int32_t, float> *C = new CRSMatrix<int32_t, float>;
    C->n_rows = 0;
    C->n_cols = 0;
    C->nnz = 0;
    C->col = nullptr;
    C->row_ptr = nullptr;
    C->val = nullptr;

    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("my_spgemm_AB", SMAX::KernelType::SPGEMM,
                          SMAX::PlatformType::CPU, SMAX::IntType::INT32,
                          SMAX::FloatType::FLOAT32);

    smax->kernel("my_spgemm_AB")
        ->register_A(A->n_rows, A->n_cols, A->nnz, A->col, A->row_ptr, A->val);
    smax->kernel("my_spgemm_AB")
        ->register_B(B->n_rows, B->n_cols, B->nnz, B->col, B->row_ptr, B->val);
    smax->kernel("my_spgemm_AB")
        ->register_C(&C->n_rows, &C->n_cols, &C->nnz, &C->col, &C->row_ptr,
                     &C->val);
    // NOTE: Since C matrix is to be generated, we need pointers to metadata,
    // not just values

    smax->kernel("my_spgemm_AB")->run();

    smax->utils->print_timers();

    C->print();

    delete A;
    delete B;
    delete C;
    delete smax;

    return 0;
}