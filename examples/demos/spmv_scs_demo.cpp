/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-vector multiplication (SpMV) using the Sell-C-sigma matrix
 * format.
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

int main(void) {
    // Initialize crs operand
    int A_crs_n_rows = 3;
    int A_crs_n_cols = 3;
    int A_crs_nnz = 5;
    int *A_crs_col = new int[A_crs_nnz]{0, 1, 1, 0, 2};
    int *A_crs_row = new int[A_crs_n_rows + 1]{0, 2, 3, 5};
    double *A_crs_val = new double[A_crs_nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    // Declare Sell-c-sigma operand
    int A_scs_C = 2;     // Defined by user
    int A_scs_sigma = 1; // Defined by user
    int A_scs_n_rows = 0;
    int A_scs_n_rows_padded = 0;
    int A_scs_n_cols = 0;
    int A_scs_n_chunks = 0;
    int A_scs_n_elements = 0;
    int A_scs_nnz = 0;
    int *A_scs_chunk_ptr = nullptr;
    int *A_scs_chunk_lengths = nullptr;
    int *A_scs_col = nullptr;
    double *A_scs_val = nullptr;
    int *A_scs_perm = nullptr;

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_scs<int, double, int>(
        A_crs_n_rows, A_crs_n_cols, A_crs_nnz, A_crs_col, A_crs_row, A_crs_val,
        A_scs_C, A_scs_sigma, A_scs_n_rows, A_scs_n_rows_padded, A_scs_n_cols,
        A_scs_n_chunks, A_scs_n_elements, A_scs_nnz, A_scs_chunk_ptr,
        A_scs_chunk_lengths, A_scs_col, A_scs_val, A_scs_perm);

    // Pad to the same length to emulate real iterative schemes
    int vec_size = std::max(A_scs_n_rows_padded, A_scs_n_cols);

    double *x = new double[vec_size];
    for (int i = 0; i < vec_size; ++i) {
        x[i] = 1.0;
    }

    // Initialize result
    double *y = new double[vec_size];

    // Register kernel tag, platform, and metadata
    smax->register_kernel("SCS_spmv", SMAX::KernelType::SPMV);

    // Register operands to this kernel tag
    // A is expected to be in the SCS format
    smax->kernel("SCS_spmv")->set_mat_scs(true);

    smax->kernel("SCS_spmv")
        ->register_A(A_scs_C, A_scs_sigma, A_scs_n_rows, A_scs_n_rows_padded,
                     A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements, A_scs_nnz,
                     A_scs_chunk_ptr, A_scs_chunk_lengths, A_scs_col, A_scs_val,
                     A_scs_perm);

    smax->kernel("SCS_spmv")->register_B(A_scs_n_rows_padded, x);
    smax->kernel("SCS_spmv")->register_C(A_scs_n_rows_padded, y);

    // Execute all phases of this kernel
    smax->kernel("SCS_spmv")->run();

    smax->utils->print_timers();

    print_vector<double>(y, A_scs_n_rows);

    delete[] A_crs_col;
    delete[] A_crs_row;
    delete[] A_crs_val;
    delete[] A_scs_chunk_ptr;
    delete[] A_scs_chunk_lengths;
    delete[] A_scs_col;
    delete[] A_scs_val;
    delete[] A_scs_perm;
    delete[] x;
    delete[] y;
    delete smax;

    return 0;
}