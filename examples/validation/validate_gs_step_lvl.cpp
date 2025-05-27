#include "../examples_common.hpp"
#include "../sptrsv_helpers.hpp"
#include "../spmv_helpers.hpp"
#include "validation_common.hpp"
#include <cmath>

void lx_difference(double *a, double *b, int num_elem, int norm, std::string descr){
    if (norm == 0) {
        double max = 0;
        for (int i = 0; i < num_elem; i++) {
            max = (max < abs(a[i] - b[i]))?abs(a[i] - b[i]):max;
        }
        std::cout << descr << " with a L-inf norm of: " << max << std::endl;
        return;
    }
    double val = 0, sum = 0;
    for (int i = 0; i < num_elem; i++) {
        val = abs(a[i] - b[i]);
        sum += pow(val, norm);
    }
    std::cout << descr << " with a L" << norm << " norm of: " << pow(sum, 1./norm) << std::endl;
}

int main(int argc, char *argv[]) {
    INIT_SPTRSV;

    DenseMatrix *x_smax = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *x_smax_perm = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *x_mkl = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *b = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *b_mkl = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *b_perm = new DenseMatrix(crs_mat->n_cols, 1, 1.0);

    int n_rows = crs_mat->n_rows;

    // Declare permutation vectors
    int *perm = new int[n_rows];
    int *inv_perm = new int[n_rows];

    // Declare and allocate room for permuted matrix
    CRSMatrix *crs_mat_perm =
        new CRSMatrix(n_rows, crs_mat->n_cols, crs_mat->nnz);

    SMAX::Interface *smax = new SMAX::Interface();
    smax->register_kernel("my_lvl_sptrsv", SMAX::KernelType::SPTRSV);
    smax->register_kernel("my_lvl_spmv", SMAX::KernelType::SPMV);

    // Generate and apply permutation
    smax->utils->generate_perm<int>(n_rows, crs_mat->row_ptr, crs_mat->col,
                                    perm, inv_perm, "JH");
    smax->utils->apply_mat_perm<int, double>(
        n_rows, crs_mat->row_ptr, crs_mat->col, crs_mat->val,
        crs_mat_perm->row_ptr, crs_mat_perm->col, crs_mat_perm->val, perm,
        inv_perm);

    smax->utils->apply_vec_perm<double>(n_rows, b->val, b_perm->val, perm);
    smax->utils->apply_vec_perm<double>(n_rows, x_smax->val, x_smax_perm->val,
                                        perm);

    CRSMatrix *crs_mat_perm_D_plus_L = new CRSMatrix;
    CRSMatrix *crs_mat_perm_U = new CRSMatrix;
    extract_D_L_U(*crs_mat_perm, *crs_mat_perm_D_plus_L, *crs_mat_perm_U);

    // Smax SPMV
    REGISTER_SPMV_DATA("my_lvl_spmv", crs_mat_perm_U, x_smax_perm, b_perm);
    smax->kernel("my_lvl_spmv")->run();
    // Smax SpTRSV
    REGISTER_SPTRSV_DATA("my_lvl_sptrsv", crs_mat_perm_D_plus_L, x_smax_perm,
                         b_perm);
    smax->kernel("my_lvl_sptrsv")->run();

    // MKL SpTRSV
    sparse_matrix_t A;
    sparse_matrix_t U;
    struct matrix_descr descr;
    struct matrix_descr descr_U;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_LOWER; // Lower triangular
    descr.diag = SPARSE_DIAG_NON_UNIT;   // Non-unit diagonal
    descr_U.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr_U.mode = SPARSE_FILL_MODE_UPPER;
    descr_U.diag = SPARSE_DIAG_NON_UNIT;

    // Create the matrix handle from CSR data
    CHECK_MKL_STATUS(mkl_sparse_d_create_csr(
                         &A, SPARSE_INDEX_BASE_ZERO, crs_mat_D_plus_L->n_rows,
                         crs_mat_D_plus_L->n_cols, crs_mat_D_plus_L->row_ptr,
                         crs_mat_D_plus_L->row_ptr + 1, crs_mat_D_plus_L->col,
                         crs_mat_D_plus_L->val),
                     "mkl_sparse_d_create_csr");
    CHECK_MKL_STATUS(mkl_sparse_d_create_csr(
                         &U, SPARSE_INDEX_BASE_ZERO, crs_mat_U->n_rows,
                         crs_mat_U->n_cols, crs_mat_U->row_ptr,
                         crs_mat_U->row_ptr + 1, crs_mat_U->col,
                         crs_mat_U->val),
                     "mkl_sparse_d_create_csr");

    // Optimize the matrix
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");
    CHECK_MKL_STATUS(mkl_sparse_optimize(U), "mkl_sparse_optimize");

    CHECK_MKL_STATUS(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, U, descr_U,
                                       x_mkl->val, 0.0, b_mkl->val),
                     "mkl_sparse_d_mv");
    CHECK_MKL_STATUS(mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                       descr, b_mkl->val, x_mkl->val),
                     "mkl_sparse_d_trsv");

    // Unpermute and compare
    smax->utils->apply_vec_perm<double>(n_rows, x_smax_perm->val, x_smax->val,
                                        inv_perm);
    smax->utils->apply_vec_perm<double>(n_rows, b_perm->val, b->val,
                                        inv_perm);

    compare_spmv(n_rows, b->val, b_mkl->val, cli_args->matrix_file_name);
    lx_difference(b_mkl->val, b->val, n_rows, 2,  "Right hand side after b = Ux");
    compare_sptrsv(n_rows, x_smax->val, x_mkl->val, cli_args->matrix_file_name);
    lx_difference(x_mkl->val, x_smax->val, n_rows, 2, "Solution after Lx=b solve");
    lx_difference(x_mkl->val, x_smax->val, n_rows, 0, "Solution after Lx=b solve");

    delete x_smax;
    delete x_mkl;
    delete b;
    delete b_mkl;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    CHECK_MKL_STATUS(mkl_sparse_destroy(U), "mkl_sparse_destroy");
    FINALIZE_SPTRSV;
}
