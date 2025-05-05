#include "../examples_common.hpp"
#include "validation_common.hpp"

int main(int argc, char *argv[]) {
    INIT_MTX;

    DenseMatrix *x = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *y_smax = new DenseMatrix(crs_mat->n_cols, 1, 0.0);
    DenseMatrix *y_mkl = new DenseMatrix(crs_mat->n_cols, 1, 0.0);

    // Smax SpMV
    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("spmv", SMAX::SPMV, SMAX::CPU);
    smax->kernels["spmv"]->register_A(crs_mat->n_rows, crs_mat->n_cols,
                                      crs_mat->nnz, &crs_mat->col,
                                      &crs_mat->row_ptr, &crs_mat->values);
    smax->kernels["spmv"]->register_B(crs_mat->n_cols, &x->values);
    smax->kernels["spmv"]->register_C(crs_mat->n_rows, &y_smax->values);

    smax->kernels["spmv"]->run();

    // MKL SpMM (SpMV with block vectors)
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Create the matrix handle from CSR data
    sparse_status_t status = mkl_sparse_d_create_csr(
        &A, SPARSE_INDEX_BASE_ZERO, crs_mat->n_rows, crs_mat->n_cols,
        crs_mat->row_ptr, crs_mat->row_ptr + 1, crs_mat->col, crs_mat->values);

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to create MKL sparse matrix.\n";
        return 1;
    }

    // Optimize the matrix
    status = mkl_sparse_optimize(A);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to optimize MKL sparse matrix.\n";
        return 1;
    }

    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr,
                             x->values, 1.0, y_mkl->values);

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "MKL sparse matrix-matrix multiply failed.\n";
        return 1;
    }

    // Compare
    compare_spmv(crs_mat->n_rows, y_smax->values, y_mkl->values,
                 cli_args->matrix_file_name);

    delete x;
    delete y_smax;
    delete y_mkl;
    mkl_sparse_destroy(A);
    DESTROY_MTX;
}