#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../validation_common.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    INIT_SPMV(IT, VT);
    IT _C = cli_args->_C;
    IT _sigma = cli_args->_sigma;

    // Declare Sell-c-sigma operand
    SCSMatrix<IT, VT> *scs_mat = new SCSMatrix<IT, VT>(_C, _sigma);

    // Smax SpMV
    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_scs<IT, VT, ULL>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, scs_mat->C, scs_mat->sigma,
        scs_mat->n_rows, scs_mat->n_rows_padded, scs_mat->n_cols,
        scs_mat->n_chunks, scs_mat->n_elements, scs_mat->nnz,
        scs_mat->chunk_ptr, scs_mat->chunk_lengths, scs_mat->col, scs_mat->val,
        scs_mat->perm);

#ifdef DEBUG_MODE
    print_vector<IT>(scs_mat->perm, scs_mat->n_rows);
#endif

    // Pad to the same length to emulate real iterative schemes
    IT vec_size = std::max(scs_mat->n_rows_padded, scs_mat->n_cols);

    DenseMatrix<VT> *x = new DenseMatrix<VT>(vec_size, 1, 1.0);
    DenseMatrix<VT> *y_smax = new DenseMatrix<VT>(vec_size, 1, 0.0);
    DenseMatrix<VT> *y_mkl = new DenseMatrix<VT>(vec_size, 1, 0.0);
    DenseMatrix<VT> *y_mkl_perm = new DenseMatrix<VT>(vec_size, 1, 0.0);

    register_kernel<IT, VT>(smax, std::string("my_scs_spmv"),
                            SMAX::KernelType::SPMV, SMAX::PlatformType::CPU);

    // A is expected to be in the SCS format
    smax->kernel("my_scs_spmv")->set_mat_scs(true);

    smax->kernel("my_scs_spmv")
        ->register_A(scs_mat->C, scs_mat->sigma, scs_mat->n_rows,
                     scs_mat->n_rows_padded, scs_mat->n_cols, scs_mat->n_chunks,
                     scs_mat->n_elements, scs_mat->nnz, scs_mat->chunk_ptr,
                     scs_mat->chunk_lengths, scs_mat->col, scs_mat->val,
                     scs_mat->perm);
    smax->kernel("my_scs_spmv")->register_B(vec_size, x->val);
    smax->kernel("my_scs_spmv")->register_C(vec_size, y_smax->val);
    smax->kernel("my_scs_spmv")->run();

    // MKL SpMV
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

    CHECK_MKL_STATUS(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A,
                                     descr, x->val, 0.0, y_mkl->val),
                     "mkl_sparse_d_mv");

    // Apply SCS permutation to result vector from MKL for consistency
    // TODO: need a standard apply_perm
    for (ULL i = 0; i < crs_mat->n_rows; ++i) {
        y_mkl_perm->val[scs_mat->perm[i]] = y_mkl->val[i];
    }

    // Compare
    compare_spmv<VT>(crs_mat->n_rows, y_smax->val, y_mkl_perm->val,
                     cli_args->matrix_file_name);

    delete x;
    delete y_smax;
    delete y_mkl;
    delete scs_mat;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPMV;
}