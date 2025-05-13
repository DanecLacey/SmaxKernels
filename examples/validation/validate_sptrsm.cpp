#include "../examples_common.hpp"
#include "../sptrsm_helpers.hpp"
#include "validation_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPTRSM;

    DenseMatrix *X_smax = new DenseMatrix(crs_mat->n_cols, n_vectors, 0.0);
    DenseMatrix *X_mkl = new DenseMatrix(crs_mat->n_cols, n_vectors, 0.0);
    DenseMatrix *B = new DenseMatrix(crs_mat->n_cols, n_vectors, 1.0);

    // Smax SpTRSM
    SMAX::Interface *smax = new SMAX::Interface();
    smax->register_kernel("my_sptrsm", SMAX::SPTRSM, SMAX::CPU);
    REGISTER_SPTRSM_DATA("my_sptrsm", crs_mat_D_plus_L, X_smax, B)
    smax->kernel("my_sptrsm")->run();

    // MKL SpTRSM
    sparse_matrix_t A;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_LOWER; // Lower triangular
    descr.diag = SPARSE_DIAG_NON_UNIT;   // Non-unit diagonal

    // Create the matrix handle from CSR data
    CHECK_MKL_STATUS(mkl_sparse_d_create_csr(
                         &A, SPARSE_INDEX_BASE_ZERO, crs_mat_D_plus_L->n_rows,
                         crs_mat_D_plus_L->n_cols, crs_mat_D_plus_L->row_ptr,
                         crs_mat_D_plus_L->row_ptr + 1, crs_mat_D_plus_L->col,
                         crs_mat_D_plus_L->val),
                     "mkl_sparse_d_create_csr");

    // Optimize the matrix
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    CHECK_MKL_STATUS(mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                       descr, SPARSE_LAYOUT_COLUMN_MAJOR,
                                       B->val, n_vectors, crs_mat->n_rows,
                                       X_mkl->val, crs_mat->n_rows),
                     "mkl_sparse_d_trsm");

    // Compare
    compare_sptrsm(crs_mat->n_rows, n_vectors, X_smax->val, X_mkl->val,
                   cli_args->matrix_file_name);

    delete X_smax;
    delete X_mkl;
    delete B;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPTRSM;
}