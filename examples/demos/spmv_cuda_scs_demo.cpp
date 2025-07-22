/**
 * @file
 * @brief Basic example demonstrating how to use the SMAX library to perform
 * sparse matrix-vector multiplication (SpMV) using the CUDA platform.
 */

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

int main(void) {
    // Initialize operands
    CRSMatrix<int, double> *A_crs = new CRSMatrix<int, double>;
    A_crs->n_rows = 3;
    A_crs->n_cols = 3;
    A_crs->nnz = 5;
    A_crs->col = new int[A->nnz]{0, 1, 1, 0, 2};
    A_crs->row_ptr = new int[A->n_rows + 1]{0, 2, 3, 5};
    A_crs->val = new double[A->nnz]{1.1, 1.2, 2.2, 3.1, 3.3};

    // Declare Sell-c-sigma operand (C = 4, sigma = 1)
    SCSMatrix<int, double> *A_scs = new SCSMatrix<int, double>(4, 1);

    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_scs<int, double, int>(
        A_crs->n_rows, A_crs->n_cols, A_crs->nnz, A_crs->col, A_crs->row_ptr,
        A_crs->val, A_scs->C, A_scs->sigma, A_scs->n_rows, A_scs->n_rows_padded,
        A_scs->n_cols, A_scs->n_chunks, A_scs->n_elements, A_scs->nnz,
        A_scs->chunk_ptr, A_scs->chunk_lengths, A_scs->col, A_scs->val,
        A_scs->perm);

    // Pad to the same length to emulate real iterative schemes
    int vec_size = std::max(A_scs->n_rows_padded, A_scs->n_cols);

    DenseMatrix<VT> *x = new DenseMatrix<VT>(A_crs->n_cols, 1, 1.0);

    // Initialize result
    DenseMatrix<VT> *y = new DenseMatrix<VT>(vec_size, 1, 0.0);

    // Register kernel tag, platform, and metadata
    smax->register_kernel("cuda_scs_spmv", SMAX::KernelType::SPMV,
                          SMAX::PlatformType::CUDA);

    // Register operands to this kernel tag
    // A is expected to be in the SCS format
    smax->kernel("cuda_scs_spmv")->set_mat_scs(true);

    smax->kernel("cuda_scs_spmv")
        ->register_A(A_scs->C, A_scs->sigma, A_scs->n_rows,
                     A_scs->n_rows_padded, A_scs->n_cols, A_scs->n_chunks,
                     A_scs->n_elements, A_scs->nnz, A_scs->chunk_ptr,
                     A_scs->chunk_lengths, A_scs->col, A_scs->val, A_scs->perm);
    // x and y are dense matrices
    smax->kernel("cuda_scs_spmv")->register_B(A_scs->n_rows_padded, x->val);
    smax->kernel("cuda_scs_spmv")->register_C(A_scs->n_rows_padded, y->val);

    // Execute all phases of this kernel
    smax->kernel("cuda_scs_spmv")->run();

    smax->utils->print_timers();

    y->print();

    delete A_crs;
    delete A_scs;
    delete x;
    delete y;
    delete smax;

    return 0;
}