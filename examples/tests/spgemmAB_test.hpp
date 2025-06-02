#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spgemmAB_test) {

    using IT = int32_t;
    using VT = float;

    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 5;
    IT *A_col = new IT[A_nnz]{0, 1, 1, 0, 2};
    IT *A_row_ptr = new IT[A_n_rows + 1]{0, 2, 3, 5};
    VT *A_val = new VT[A_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    int B_n_rows = 3;
    int B_n_cols = 3;
    int B_nnz = 5;
    IT *B_col = new IT[A_nnz]{0, 2, 0, 1, 0};
    IT *B_row_ptr = new IT[A_n_rows + 1]{0, 2, 4, 5};
    VT *B_val = new VT[A_nnz]{1.1, 1.3, 2.1, 2.2, 3.1};

    int C_n_rows = 0;
    int C_n_cols = 0;
    int C_nnz = 0;
    IT *C_col = nullptr;
    IT *C_row_ptr = nullptr;
    VT *C_val = nullptr;

    int expected_C_n_rows = 3;
    int expected_C_n_cols = 3;
    int expected_C_nnz = 7;
    IT *expected_C_col = new IT[expected_C_nnz]{0, 2, 1, 0, 1, 0, 2};
    IT *expected_C_row_ptr = new IT[expected_C_n_rows + 1]{0, 3, 5, 7};
    VT *expected_C_val = new VT[expected_C_nnz]{
        3.73000001907348633, 1.42999994754791260, 2.64000010490417480,
        4.61999988555908203, 4.84000015258789062, 13.63999938964843750,
        4.02999973297119141};

    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("my_spgemm_AB", SMAX::KernelType::SPGEMM,
                          SMAX::PlatformType::CPU, SMAX::IntType::INT32,
                          SMAX::FloatType::FLOAT32);

    smax->kernel("my_spgemm_AB")
        ->register_A(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);
    smax->kernel("my_spgemm_AB")
        ->register_B(B_n_rows, B_n_cols, B_nnz, B_col, B_row_ptr, B_val);
    smax->kernel("my_spgemm_AB")
        ->register_C(&C_n_rows, &C_n_cols, &C_nnz, &C_col, &C_row_ptr, &C_val);

    // Function to test
    smax->kernel("my_spgemm_AB")->run();

    compare_values<int>(expected_C_n_rows, C_n_rows, std::string("n_rows"));
    compare_values<int>(expected_C_n_cols, C_n_cols, std::string("n_cols"));
    compare_values<int>(expected_C_nnz, C_nnz, std::string("nnz"));
    compare_arrays<IT>(expected_C_col, C_col, C_nnz, std::string("col"));
    compare_arrays<IT>(expected_C_row_ptr, C_row_ptr, C_n_rows + 1,
                       std::string("row_ptr"));
    compare_arrays<VT>(expected_C_val, C_val, C_nnz, std::string("val"));

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] B_col;
    delete[] B_row_ptr;
    delete[] B_val;
    delete[] C_col;
    delete[] C_row_ptr;
    delete[] C_val;
    delete[] expected_C_col;
    delete[] expected_C_row_ptr;
    delete[] expected_C_val;
    delete smax;
}
