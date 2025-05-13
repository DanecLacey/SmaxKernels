#include "../examples_common.hpp"
#include "../sptrsv_helpers.hpp"
#include "validation_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPTRSV;

    DenseMatrix *x_smax = new DenseMatrix(crs_mat->n_cols, 1, 0.0);
    DenseMatrix *x_mkl = new DenseMatrix(crs_mat->n_cols, 1, 0.0);
    DenseMatrix *b = new DenseMatrix(crs_mat->n_cols, 1, 1.0);

    // Smax SpTRSV
    SMAX::Interface *smax = new SMAX::Interface();
    smax->register_kernel("my_sptrsv", SMAX::SPTRSV, SMAX::CPU);
    REGISTER_SPTRSV_DATA("my_sptrsv", crs_mat_D_plus_L, x_smax, b)
    smax->kernel("my_sptrsv")->run();

    // MKL SpTRSV
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

    CHECK_MKL_STATUS(mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                       descr, b->val, x_mkl->val),
                     "mkl_sparse_d_trsv");

    // Compare
    compare_sptrsv(crs_mat->n_rows, x_smax->val, x_mkl->val,
                   cli_args->matrix_file_name);

    delete x_smax;
    delete x_mkl;
    delete b;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPTRSV;
}