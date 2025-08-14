#include "../../examples_common.hpp"
#include "../../sptrsm_helpers.hpp"
#include "../validation_common.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    INIT_SPTRSM(IT, VT);

    DenseMatrix<VT> *X_smax =
        new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 0.0);
    DenseMatrix<VT> *X_mkl =
        new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 0.0);
    DenseMatrix<VT> *B = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 1.0);

    // Smax SpTRSM
    SMAX::Interface *smax = new SMAX::Interface();
    register_kernel<IT, VT>(smax, std::string("my_sptrsm"),
                            SMAX::KernelType::SPTRSM, SMAX::PlatformType::CPU);
    REGISTER_SPTRSM_DATA("my_sptrsm", crs_mat_D_plus_L, X_smax, B)
    smax->kernel("my_sptrsm")->run();

    // MKL SpTRSM
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

    CHECK_MKL_STATUS(mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                       descr, SPARSE_LAYOUT_COLUMN_MAJOR,
                                       B->val, n_vectors, crs_mat->n_rows,
                                       X_mkl->val, crs_mat->n_rows),
                     "mkl_sparse_d_trsm");

    // Compare
    compare_sptrsm<VT>(crs_mat->n_rows, n_vectors, X_smax->val, X_mkl->val,
                       cli_args->matrix_file_name);

    delete X_smax;
    delete X_mkl;
    delete B;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPTRSM;
}