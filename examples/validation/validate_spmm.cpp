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
    REGISTER_SPMM_KERNEL("my_spmm", crs_mat, n_vectors, X, Y_smax);
    smax->kernels["my_spmm"]->run();

    // MKL SpMM
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

    status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                             1.0, // alpha
                             A, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X->values,
                             n_vectors,
                             crs_mat->n_cols, // leading dimension of X
                             0.0,             // beta
                             Y_mkl->values,
                             crs_mat->n_rows // leading dimension of Y
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "MKL sparse matrix-matrix multiply failed.\n";
        return 1;
    }

    // Compare
    compare_spmm(crs_mat->n_rows, n_vectors, Y_smax->values, Y_mkl->values,
                 cli_args->matrix_file_name);

    delete X;
    delete Y_smax;
    delete Y_mkl;
    mkl_sparse_destroy(A);
    FINALIZE_SPMM;
}