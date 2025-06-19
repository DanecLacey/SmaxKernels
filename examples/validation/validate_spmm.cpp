#include "../examples_common.hpp"
#include "../spmm_helpers.hpp"
#include "validation_common.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    INIT_SPMM(IT, VT);

    DenseMatrix<VT> *X = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix<VT> *Y_smax =
        new DenseMatrix<VT>(crs_mat->n_rows, n_vectors, 0.0);
    DenseMatrix<VT> *Y_mkl =
        new DenseMatrix<VT>(crs_mat->n_rows, n_vectors, 0.0);

    // Smax SpMM
    SMAX::Interface *smax = new SMAX::Interface();
    register_kernel<IT, VT>(smax, std::string("my_spmm"),
                            SMAX::KernelType::SPMM, SMAX::PlatformType::CPU);
    REGISTER_SPMM_DATA("my_spmm", crs_mat, n_vectors, X, Y_smax);
    smax->kernel("my_spmm")->run();

    // MKL SpMM
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Create the matrix handle from CSR data
    CHECK_MKL_STATUS(
        mkl_sparse_d_create_csr(
            /* handle    */ &A,
            /* indexing  */ SPARSE_INDEX_BASE_ZERO,
            /* rows      */ static_cast<MKL_INT>(crs_mat->n_rows),
            /* cols      */ static_cast<MKL_INT>(crs_mat->n_cols),
            /* row_start */ reinterpret_cast<MKL_INT *>(crs_mat->row_ptr),
            /* row_end   */ reinterpret_cast<MKL_INT *>(crs_mat->row_ptr + 1),
            /* col_ind   */ reinterpret_cast<MKL_INT *>(crs_mat->col),
            /* values    */ crs_mat->val),
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
    compare_spmm<VT>(crs_mat->n_rows, n_vectors, Y_smax->val, Y_mkl->val,
                     cli_args->matrix_file_name);

    delete X;
    delete Y_smax;
    delete Y_mkl;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPMM;
}
