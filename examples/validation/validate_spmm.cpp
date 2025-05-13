#include "../examples_common.hpp"
#include "../spmm_helpers.hpp"
#include "validation_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPMM;

    DenseMatrix *X = new DenseMatrix(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix *Y_smax = new DenseMatrix(crs_mat->n_cols, n_vectors, 0.0);
    DenseMatrix *Y_mkl = new DenseMatrix(crs_mat->n_cols, n_vectors, 0.0);

    // Smax SpMM
    SMAX::Interface *smax = new SMAX::Interface();
    smax->register_kernel("my_spmm", SMAX::SPMM, SMAX::CPU);
    REGISTER_SPMM_DATA("my_spmm", crs_mat, n_vectors, X, Y_smax);
    smax->kernel("my_spmm")->run();

    // MKL SpMM
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Create the matrix handle from CSR data
    CHECK_MKL_STATUS(mkl_sparse_d_create_csr(
                         &A, SPARSE_INDEX_BASE_ZERO, crs_mat->n_rows,
                         crs_mat->n_cols, crs_mat->row_ptr,
                         crs_mat->row_ptr + 1, crs_mat->col, crs_mat->val),
                     "mkl_sparse_d_create_csr");

    // Optimize the matrix
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    CHECK_MKL_STATUS(mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                     1.0, // alpha
                                     A, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
                                     X->val, n_vectors,
                                     crs_mat->n_cols, // leading dimension of X
                                     0.0,             // beta
                                     Y_mkl->val,
                                     crs_mat->n_rows // leading dimension of Y
                                     ),
                     "mkl_sparse_d_mm");

    // Compare
    compare_spmm(crs_mat->n_rows, n_vectors, Y_smax->val, Y_mkl->val,
                 cli_args->matrix_file_name);

    delete X;
    delete Y_smax;
    delete Y_mkl;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPMM;
}