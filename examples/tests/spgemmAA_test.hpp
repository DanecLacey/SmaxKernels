#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spgemmAA_test) {

    using IT = uint16_t;
    using VT = float;

    CRSMatrix<IT, VT> *A = new CRSMatrix<IT, VT>(3, 3, 5);
    A->col = new IT[A->nnz]{0, 1, 1, 0, 2};
    A->row_ptr = new IT[A->n_rows + 1]{0, 2, 3, 5};
    A->val = new VT[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    CRSMatrix<IT, VT> *C = new CRSMatrix<IT, VT>;

    CRSMatrix<IT, VT> *C_exp = new CRSMatrix<IT, VT>(3, 3, 6);
    C_exp->col = new IT[C_exp->nnz]{0, 1, 1, 0, 1, 2};
    C_exp->row_ptr = new IT[C_exp->n_rows + 1]{0, 2, 3, 6};
    C_exp->val = new VT[C_exp->nnz]{1.21000003814697266, 3.96000027656555176,
                                    4.84000015258789062, 13.63999938964843750,
                                    3.72000002861022949, 10.88999938964843750};

    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("spgemm_AA", SMAX::KernelType::SPGEMM,
                          SMAX::PlatformType::CPU, SMAX::IntType::UINT16,
                          SMAX::FloatType::FLOAT32);

    smax->kernel("spgemm_AA")
        ->register_A(A->n_rows, A->n_cols, A->nnz, A->col, A->row_ptr, A->val);
    smax->kernel("spgemm_AA")
        ->register_B(A->n_rows, A->n_cols, A->nnz, A->col, A->row_ptr, A->val);
    smax->kernel("spgemm_AA")
        ->register_C(&C->n_rows, &C->n_cols, &C->nnz, &C->col, &C->row_ptr,
                     &C->val);

    // Function to test
    smax->kernel("spgemm_AA")->run();

    compare_values<ULL>(C_exp->n_rows, C->n_rows, std::string("n_rows"));
    compare_values<ULL>(C_exp->n_cols, C->n_cols, std::string("n_cols"));
    compare_values<ULL>(C_exp->nnz, C->nnz, std::string("nnz"));
    compare_arrays<IT>(C_exp->col, C->col, C->nnz, std::string("col"));
    compare_arrays<IT>(C_exp->row_ptr, C->row_ptr, C->n_rows + 1,
                       std::string("row_ptr"));
    compare_arrays<VT>(C_exp->val, C->val, C->nnz, std::string("val"));

    delete A;
    delete C;
    delete C_exp;
    delete smax;
}
