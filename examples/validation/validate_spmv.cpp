#include "../examples_common.hpp"
#include "../spmv_helpers.hpp"
#include "validation_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPMV;
    DenseMatrix *x = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *y_smax = new DenseMatrix(crs_mat->n_cols, 1, 0.0);
    DenseMatrix *y_mkl = new DenseMatrix(crs_mat->n_cols, 1, 0.0);

    // Smax SpMV
    SMAX::Interface *smax = new SMAX::Interface();
    smax->register_kernel("my_spmv", SMAX::KernelType::SPMV);
    REGISTER_SPMV_DATA("my_spmv", crs_mat, x, y_smax);
    smax->kernel("my_spmv")->run();

    // MKL SpMV
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

    CHECK_MKL_STATUS(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                     descr, x->val, 0.0, y_mkl->val),
                     "mkl_sparse_d_mv");

    // Compare
    compare_spmv(crs_mat->n_rows, y_smax->val, y_mkl->val,
                 cli_args->matrix_file_name);

    delete x;
    delete y_smax;
    delete y_mkl;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPMV;
}