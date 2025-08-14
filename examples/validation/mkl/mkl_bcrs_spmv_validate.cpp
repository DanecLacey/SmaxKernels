#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../validation_common.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    INIT_SPMV(IT, VT);
    IT hpad = cli_args->_hpad;
    IT wpad = cli_args->_wpad;
    bool use_cm = cli_args->_use_cm;

    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *y_smax = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);
    DenseMatrix<VT> *y_mkl = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

    // Smax SpMV
    SMAX::Interface *smax = new SMAX::Interface();

    register_kernel<IT, VT>(smax, std::string("my_bcrs_spmv"),
                            SMAX::KernelType::SPMV, SMAX::PlatformType::CPU);

    // Declare bcrs operand
    BCRSMatrix<IT, VT> *A_bcrs = new BCRSMatrix<IT, VT>();

    // Convert CRS matrix to BCRS
    smax->utils->convert_crs_to_bcrs<IT, VT, ULL>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, A_bcrs->n_rows, A_bcrs->n_cols,
        A_bcrs->n_blocks, A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
        A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr, A_bcrs->val, hpad, wpad,
        hpad, wpad, use_cm);

    smax->kernel("my_bcrs_spmv")->set_mat_bcrs(true);
    smax->kernel("my_bcrs_spmv")->set_block_column_major(use_cm);

    // A is assumed to be in BCRS format
    smax->kernel("my_bcrs_spmv")
        ->register_A(A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
                     A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
                     A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr,
                     A_bcrs->val);
    // x and y are dense matrices
    smax->kernel("my_bcrs_spmv")->register_B(A_bcrs->n_cols, x->val);
    smax->kernel("my_bcrs_spmv")->register_C(A_bcrs->n_rows, y_smax->val);

    smax->kernel("my_bcrs_spmv")->run();

    // MKL SpMV
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Create the matrix handle from CSR data
    CHECK_MKL_STATUS(
        mkl_sparse_d_create_csr(
            /* handle    */ &A,
            /* indexing  */ SPARSE_INDEX_BASE_ZERO,
            /* rows      */ static_cast<MKL_INT>(crs_mat->n_rows),
            /* cols      */ static_cast<MKL_INT>(crs_mat->n_cols),
            /* row_start */ reinterpret_cast<MKL_INT *>(crs_mat->row_ptr),
            /* row_end   */
            reinterpret_cast<MKL_INT *>(crs_mat->row_ptr + 1),
            /* col_ind   */ reinterpret_cast<MKL_INT *>(crs_mat->col),
            /* values    */ crs_mat->val),
        "mkl_sparse_d_create_csr");

    // Optimize the matrix
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    CHECK_MKL_STATUS(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                     descr, x->val, 0.0, y_mkl->val),
                     "mkl_sparse_d_mv");

    // Compare
    compare_spmv<VT>(crs_mat->n_rows, y_smax->val, y_mkl->val,
                     cli_args->matrix_file_name);

    delete x;
    delete y_smax;
    delete y_mkl;
    delete A_bcrs;

    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPMV;
}