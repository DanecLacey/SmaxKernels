#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../validation_common.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    INIT_SPTRSV_LVL(IT, VT);
    DenseMatrix<VT> *x_smax = new DenseMatrix<VT>(crs_mat->n_cols, 1, 0.0);
    DenseMatrix<VT> *x_smax_perm = new DenseMatrix<VT>(crs_mat->n_cols, 1, 0.0);
    DenseMatrix<VT> *x_mkl = new DenseMatrix<VT>(crs_mat->n_cols, 1, 0.0);
    DenseMatrix<VT> *b = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *b_perm = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);

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

    CRSMatrix<IT, VT> *crs_mat_D_plus_L = new CRSMatrix<IT, VT>;
    CRSMatrix<IT, VT> *crs_mat_U = new CRSMatrix<IT, VT>;
    extract_D_L_U<IT, VT>(*crs_mat_perm, *crs_mat_D_plus_L, *crs_mat_U);

    // Smax SpTRSV
    REGISTER_SPTRSV_DATA("my_lvl_sptrsv", crs_mat_D_plus_L, x_smax_perm,
                         b_perm);
    smax->kernel("my_lvl_sptrsv")->set_mat_perm(true);
    smax->kernel("my_lvl_sptrsv")->run();

    // MKL SpTRSV
    sparse_matrix_t A;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_LOWER; // Lower triangular
    descr.diag = SPARSE_DIAG_NON_UNIT;   // Non-unit diagonal

    // Create the matrix handle from CSR data
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

    // Optimize the matrix
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    CHECK_MKL_STATUS(mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                       descr, b->val, x_mkl->val),
                     "mkl_sparse_d_trsv");

    // Unpermute and compare
    smax->utils->apply_vec_perm<VT>(n_rows, x_smax_perm->val, x_smax->val,
                                    inv_perm);

    compare_sptrsv<VT>(n_rows, x_smax->val, x_mkl->val,
                       cli_args->matrix_file_name);

    delete x_smax;
    delete x_mkl;
    delete b;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPTRSV;
}
