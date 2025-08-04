#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"

int main(void) {
    // Initialize crs operand
    CRSMatrix<int, double> *A_crs = new CRSMatrix<int, double>;
    A_crs->n_rows = 4;
    A_crs->n_cols = 4;
    A_crs->nnz = 5;
    A_crs->col = new int[A_crs->nnz]{0, 2, 3, 1, 3};
    A_crs->row_ptr = new int[A_crs->n_rows + 1]{0, 3, 4, 4, 5};
    A_crs->val = new double[A_crs->nnz]{1.1, 1.3, 1.4, 2.2, 4.4};

    // Declare bcrs operand
    BCRSMatrix<int, double> *A_bcrs = new BCRSMatrix<int, double>;
    const bool use_blocked_column_major = true;

    DenseMatrix<double> *x = new DenseMatrix<double>(A_crs->n_cols, 1, 1.0);

    // Initialize result
    DenseMatrix<double> *y = new DenseMatrix<double>(A_crs->n_rows, 1, 0.0);

    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_bcrs<int, double, ULL>(
        A_crs->n_rows, A_crs->n_cols, A_crs->nnz, A_crs->col, A_crs->row_ptr,
        A_crs->val, A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
        A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad, A_bcrs->b_w_pad,
        A_bcrs->col, A_bcrs->row_ptr, A_bcrs->val, 2, 2, 2, 2,
        use_blocked_column_major);

    smax->register_kernel("my_bcrs_spmv", SMAX::KernelType::SPMV);

    // A is expected to be in the BCRS format
    smax->kernel("my_bcrs_spmv")->set_mat_bcrs(true);

    smax->kernel("my_bcrs_spmv")
        ->set_block_column_major(use_blocked_column_major);

    // Register operands to this kernel tag
    // A is assumed to be in CRS format
    smax->kernel("my_bcrs_spmv")
        ->register_A(A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
                     A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
                     A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr,
                     A_bcrs->val);

    // x and y are dense matrices
    smax->kernel("my_bcrs_spmv")->register_B(A_bcrs->n_cols, x->val);
    smax->kernel("my_bcrs_spmv")->register_C(A_bcrs->n_rows, y->val);

    // Execute all phases of this kernel
    smax->kernel("my_bcrs_spmv")->run();

    smax->utils->print_timers();

    y->print();

    delete A_crs;
    delete A_bcrs;
    delete x;
    delete y;
    delete smax;

    return 0;
}