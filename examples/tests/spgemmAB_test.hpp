#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spgemmAB_test) {

    using IT = int32_t;
    using VT = float;

    CRSMatrix<IT, VT> *A = new CRSMatrix<IT, VT>(3, 3, 5);
    A->col = new IT[A->nnz]{0, 1, 1, 0, 2};
    A->row_ptr = new IT[A->n_rows + 1]{0, 2, 3, 5};
    A->val = new VT[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    CRSMatrix<IT, VT> *B = new CRSMatrix<IT, VT>(3, 3, 5);
    B->col = new IT[B->nnz]{0, 2, 0, 1, 0};
    B->row_ptr = new IT[B->n_rows + 1]{0, 2, 4, 5};
    B->val = new VT[B->nnz]{1.1, 1.3, 2.1, 2.2, 3.1};

    CRSMatrix<IT, VT> *C = new CRSMatrix<IT, VT>;

    CRSMatrix<IT, VT> *C_exp = new CRSMatrix<IT, VT>(3, 3, 7);
    C_exp->col = new IT[C_exp->nnz]{0, 2, 1, 0, 1, 0, 2};
    C_exp->row_ptr = new IT[C_exp->n_rows + 1]{0, 3, 5, 7};
    C_exp->val = new VT[C_exp->nnz]{3.73000001907348633, 1.42999994754791260,
                                    2.64000010490417480, 4.61999988555908203,
                                    4.84000015258789062, 13.63999938964843750,
                                    4.02999973297119141};

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

    // Function to test
    smax->kernel("my_spgemm_AB")->run();

    compare_values<ULL>(C_exp->n_rows, C->n_rows, std::string("n_rows"));
    compare_values<ULL>(C_exp->n_cols, C->n_cols, std::string("n_cols"));
    compare_values<ULL>(C_exp->nnz, C->nnz, std::string("nnz"));
    compare_arrays<IT>(C_exp->col, C->col, C->nnz, std::string("col"));
    compare_arrays<IT>(C_exp->row_ptr, C->row_ptr, C->n_rows + 1,
                       std::string("row_ptr"));
    compare_arrays<VT>(C_exp->val, C->val, C->nnz, std::string("val"));

    delete A;
    delete B;
    delete C;
    delete C_exp;
    delete smax;
}
