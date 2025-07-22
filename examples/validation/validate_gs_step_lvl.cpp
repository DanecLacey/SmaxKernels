#include "../examples_common.hpp"
#include "../gs_helpers.hpp"
#include "validation_common.hpp"
#include <cmath>
#include <iostream>

template <typename VT>
void lx_difference(VT *a, VT *b, int num_elem, int norm, std::string descr) {
    if (norm == 0) {
        double max = 0;
        for (int i = 0; i < num_elem; i++) {
            max = (max < abs(a[i] - b[i])) ? abs(a[i] - b[i]) : max;
        }
        std::cout << descr << " with a L-inf norm of: " << max << std::endl;
        return;
    }
    double val = 0, sum = 0;
    for (int i = 0; i < num_elem; i++) {
        val = abs(a[i] - b[i]);
        sum += pow(val, norm);
    }
    std::cout << descr << " with a L" << norm
              << " norm of: " << pow(sum, 1. / norm) << std::endl;
}

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    INIT_GS(IT, VT);
    CRSMatrix<IT, VT> *crs_mat_D_plus_L = new CRSMatrix<IT, VT>;
    CRSMatrix<IT, VT> *crs_mat_U = new CRSMatrix<IT, VT>;
    extract_D_L_U<IT, VT>(*crs_mat, *crs_mat_D_plus_L, *crs_mat_U);
    DenseMatrix<VT> *x_smax = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *x_smax_perm = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *x_mkl = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *b = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *b_mkl = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *b_perm = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *tmp_rhs = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *tmp_rhs_mkl = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);

    ULL n_rows = crs_mat->n_rows;

    // Declare permutation vectors
    int *perm = new int[n_rows];
    int *inv_perm = new int[n_rows];

    // Declare and allocate room for permuted matrix
    CRSMatrix<IT, VT> *crs_mat_perm =
        new CRSMatrix<IT, VT>(n_rows, crs_mat->n_cols, crs_mat->nnz);

    SMAX::Interface *smax = new SMAX::Interface();
    register_kernel<IT, VT>(smax, std::string("my_lvl_sptrsv"),
                            SMAX::KernelType::SPTRSV, SMAX::PlatformType::CPU);
    register_kernel<IT, VT>(smax, std::string("my_lvl_spmv"),
                            SMAX::KernelType::SPMV, SMAX::PlatformType::CPU);

    // Generate and apply permutation
    smax->utils->generate_perm<IT>(n_rows, crs_mat->row_ptr, crs_mat->col, perm,
                                   inv_perm, argv[2]);
    smax->utils->apply_mat_perm<IT, VT>(n_rows, crs_mat->row_ptr, crs_mat->col,
                                        crs_mat->val, crs_mat_perm->row_ptr,
                                        crs_mat_perm->col, crs_mat_perm->val,
                                        perm, inv_perm);

    smax->utils->apply_vec_perm<VT>(n_rows, b->val, b_perm->val, perm);
    smax->utils->apply_vec_perm<VT>(n_rows, x_smax->val, x_smax_perm->val,
                                    perm);

    CRSMatrix<IT, VT> *crs_mat_perm_D_plus_L = new CRSMatrix<IT, VT>;
    CRSMatrix<IT, VT> *crs_mat_perm_U = new CRSMatrix<IT, VT>;
    extract_D_L_U<IT, VT>(*crs_mat_perm, *crs_mat_perm_D_plus_L,
                          *crs_mat_perm_U);

    // Smax SPMV
    REGISTER_GS_DATA("my_lvl_spmv", crs_mat_perm_U, x_smax_perm, tmp_rhs);
    smax->kernel("my_lvl_spmv")->run();
    // Smax b - Ux
    *b_perm -= *tmp_rhs;
    // Smax SpTRSV
    REGISTER_GS_DATA("my_lvl_sptrsv", crs_mat_perm_D_plus_L, x_smax_perm,
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

    // // Create the matrix handle from CSR data
    CHECK_MKL_STATUS(
        mkl_sparse_d_create_csr(
            /* handle    */ &A,
            /* indexing  */ SPARSE_INDEX_BASE_ZERO,
            /* rows      */ static_cast<MKL_INT>(crs_mat_D_plus_L->n_rows),
            /* cols      */ static_cast<MKL_INT>(crs_mat_D_plus_L->n_cols),
            /* row_start */
            reinterpret_cast<MKL_INT *>(crs_mat_D_plus_L->row_ptr),
            /* row_end   */
            reinterpret_cast<MKL_INT *>(crs_mat_D_plus_L->row_ptr + 1),
            /* col_ind   */ reinterpret_cast<MKL_INT *>(crs_mat_D_plus_L->col),
            /* values    */ crs_mat_D_plus_L->val),
        "mkl_sparse_d_create_csr");

    CHECK_MKL_STATUS(
        mkl_sparse_d_create_csr(
            /* handle    */ &U,
            /* indexing  */ SPARSE_INDEX_BASE_ZERO,
            /* rows      */ static_cast<MKL_INT>(crs_mat_U->n_rows),
            /* cols      */ static_cast<MKL_INT>(crs_mat_U->n_cols),
            /* row_start */ reinterpret_cast<MKL_INT *>(crs_mat_U->row_ptr),
            /* row_end   */ reinterpret_cast<MKL_INT *>(crs_mat_U->row_ptr + 1),
            /* col_ind   */ reinterpret_cast<MKL_INT *>(crs_mat_U->col),
            /* values    */ crs_mat_U->val),
        "mkl_sparse_d_create_csr");

    // Optimize the matrix
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");
    CHECK_MKL_STATUS(mkl_sparse_optimize(U), "mkl_sparse_optimize");

    CHECK_MKL_STATUS(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, U,
                                     descr_U, x_mkl->val, 0.0,
                                     tmp_rhs_mkl->val),
                     "mkl_sparse_d_mv");
    *b_mkl -= *tmp_rhs_mkl;
    CHECK_MKL_STATUS(mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                       descr, b_mkl->val, x_mkl->val),
                     "mkl_sparse_d_trsv");

    // Unpermute and compare
    smax->utils->apply_vec_perm<VT>(n_rows, x_smax_perm->val, x_smax->val,
                                    inv_perm);
    smax->utils->apply_vec_perm<VT>(n_rows, b_perm->val, b->val, inv_perm);

    compare_gs<VT>(n_rows, b->val, b_mkl->val, cli_args->matrix_file_name);
    lx_difference<VT>(b_mkl->val, b->val, n_rows, 2,
                      "Right hand side after b = Ux");
    compare_gs<VT>(n_rows, x_smax->val, x_mkl->val, cli_args->matrix_file_name);
    lx_difference<VT>(x_mkl->val, x_smax->val, n_rows, 2,
                      "Solution after Lx=b solve");
    lx_difference<VT>(x_mkl->val, x_smax->val, n_rows, 0,
                      "Solution after Lx=b solve");

    delete x_smax;
    delete x_smax_perm;
    delete x_mkl;
    delete b;
    delete b_mkl;
    delete b_perm;
    delete tmp_rhs;
    delete tmp_rhs_mkl;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    CHECK_MKL_STATUS(mkl_sparse_destroy(U), "mkl_sparse_destroy");
    FINALIZE_GS;
}
