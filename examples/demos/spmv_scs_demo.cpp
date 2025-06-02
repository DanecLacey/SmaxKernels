/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-vector multiplication (SpMV) using the Sell-C-sigma matrix
 * format.
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

int main(void) {
    // // Initialize COO operand
    // int A_coo_n_rows = 3;
    // int A_coo_n_cols = 3;
    // int A_coo_nnz = 5;
    // int *A_coo_col = new int[A_coo_nnz]{0, 1, 1, 0, 2};
    // int *A_coo_row = new int[A_coo_n_rows + 1]{0, 2, 3, 5};
    // double *A_coo_val = new double[A_coo_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    // // Declare Sell-c-sigma operand
    // int A_scs_C = 1;     // Defined by user
    // int A_scs_sigma = 1; // Defined by user
    // int A_scs_n_rows = 0;
    // int A_scs_n_rows_padded = 0;
    // int A_scs_n_cols = 0;
    // int A_scs_n_chunks = 0;
    // int A_scs_n_elements = 0;
    // int A_scs_nnz = 0;
    // int *A_scs_chunk_ptrs = nullptr;
    // int *A_scs_chunk_lengths = nullptr;
    // int *A_scs_col = nullptr;
    // double *A_scs_val = nullptr;
    // int *A_scs_perm = nullptr;
    // int *A_scs_inv_perm = nullptr;

    // double *x = new double[A_coo_n_cols];
    // for (int i = 0; i < A_coo_n_cols; ++i) {
    //     x[i] = 1.0;
    // }

    // // Initialize result
    // double *y = new double[A_coo_n_rows];

    // // Initialize interface object
    // SMAX::Interface *smax = new SMAX::Interface();

    // smax->utils->convert_coo_to_scs<int, double>(
    //     A_coo_n_rows, A_coo_n_cols, A_coo_nnz, A_coo_col, A_coo_row,
    //     A_coo_val, A_scs_C, A_scs_sigma, A_scs_n_rows, A_scs_n_rows_padded,
    //     A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements, A_scs_nnz,
    //     A_scs_chunk_ptrs, A_scs_chunk_lengths, A_scs_col, A_scs_val,
    //     A_scs_perm, A_scs_inv_perm);

    // // Register kernel tag, platform, and metadata
    // smax->register_kernel("SCS_spmv", SMAX::KernelType::SPMV);

    // // Register operands to this kernel tag
    // // A is expected to be in the SCS format
    // smax->kernel("SCS_spmv")->set_mat_scs(true);

    // smax->kernel("SCS_spmv")
    //     ->register_A(A_scs_n_rows, A_scs_n_rows_padded, A_scs_n_cols,
    //     A_scs_C,
    //                  A_scs_sigma, A_scs_n_chunks, A_scs_n_elements,
    //                  A_scs_nnz, A_scs_chunk_ptrs, A_scs_chunk_lengths,
    //                  A_scs_col, A_scs_val, A_scs_perm, A_scs_inv_perm);
    // smax->kernel("SCS_spmv")->register_B(A_scs_n_cols, x);
    // smax->kernel("SCS_spmv")->register_C(A_scs_n_rows, y);

    // // Execute all phases of this kernel
    // smax->kernel("SCS_spmv")->run();

    // smax->utils->print_timers();

    // print_vector<double>(y, A_scs_n_cols);

    // delete[] A_coo_col;
    // delete[] A_coo_row;
    // delete[] A_coo_val;
    // delete[] A_scs_chunk_ptrs;
    // delete[] A_scs_chunk_lengths;
    // delete[] A_scs_col;
    // delete[] A_scs_val;
    // delete[] A_scs_perm;
    // delete[] A_scs_inv_perm;
    // delete[] x;
    // delete[] y;
    // delete smax;

    // return 0;
}