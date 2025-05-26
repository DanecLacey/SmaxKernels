#pragma once

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "testing_framework.hpp"
#include "tests_common.hpp"

REGISTER_TEST(spgemmAA_test) {

#ifdef _OPENMP
    omp_set_num_threads(1);
#endif

    using IT = uint16_t;
    using VT = float;

    IT A_n_rows = 3;
    IT A_n_cols = 3;
    IT A_nnz = 5;
    IT *A_col = new IT[A_nnz]{0, 1, 1, 0, 2};
    IT *A_row_ptr = new IT[A_n_rows + 1]{0, 2, 3, 5};
    VT *A_val = new VT[A_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    IT C_n_rows = 0;
    IT C_n_cols = 0;
    IT C_nnz = 0;
    IT *C_col = nullptr;
    IT *C_row_ptr = nullptr;
    VT *C_val = nullptr;

    IT expected_C_n_rows = 3;
    IT expected_C_n_cols = 3;
    IT expected_C_nnz = 6;
    IT *expected_C_col = new IT[expected_C_nnz]{0, 1, 1, 0, 1, 2};
    IT *expected_C_row_ptr = new IT[expected_C_n_rows + 1]{0, 2, 3, 6};
    VT *expected_C_val = new VT[expected_C_nnz]{
        1.21000003814697266,  3.96000027656555176, 4.84000015258789062,
        13.63999938964843750, 3.72000002861022949, 10.88999938964843750};

    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("spgemm_AA", SMAX::KernelType::SPGEMM,
                          SMAX::PlatformType::CPU, SMAX::IntType::UINT16,
                          SMAX::FloatType::FLOAT32);

    smax->kernel("spgemm_AA")
        ->register_A(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);
    smax->kernel("spgemm_AA")
        ->register_B(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);
    smax->kernel("spgemm_AA")
        ->register_C(&C_n_rows, &C_n_cols, &C_nnz, &C_col, &C_row_ptr, &C_val);

    // Function to test
    smax->kernel("spgemm_AA")->run();

    compare_values<IT>(expected_C_n_rows, C_n_rows, std::string("n_rows"));
    compare_values<IT>(expected_C_n_cols, C_n_cols, std::string("n_cols"));
    compare_values<IT>(expected_C_nnz, C_nnz, std::string("nnz"));
    compare_arrays<IT>(expected_C_col, C_col, C_nnz, std::string("col"));
    compare_arrays<IT>(expected_C_row_ptr, C_row_ptr, C_n_rows + 1,
                       std::string("row_ptr"));
    compare_arrays<VT>(expected_C_val, C_val, C_nnz, std::string("val"));

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] C_col;
    delete[] C_row_ptr;
    delete[] C_val;
    delete[] expected_C_col;
    delete[] expected_C_row_ptr;
    delete[] expected_C_val;
    delete smax;
}
