#include "../examples_common.hpp"
#include "../spmv_helpers.hpp"
#include "validation_common.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    INIT_SPMV(IT, VT);

    // TODO: Make C and sigma runtime args
    // Declare Sell-c-sigma operand
    int A_scs_C = 4;      // Defined by user
    int A_scs_sigma = 16; // Defined by user
    int A_scs_n_rows = 0;
    int A_scs_n_rows_padded = 0;
    int A_scs_n_cols = 0;
    int A_scs_n_chunks = 0;
    int A_scs_n_elements = 0;
    int A_scs_nnz = 0;
    IT *A_scs_chunk_ptr = nullptr;
    IT *A_scs_chunk_lengths = nullptr;
    IT *A_scs_col = nullptr;
    VT *A_scs_val = nullptr;
    IT *A_scs_perm = nullptr;

    // Smax SpMV
    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->convert_crs_to_scs<IT, VT, int>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, A_scs_C, A_scs_sigma, A_scs_n_rows,
        A_scs_n_rows_padded, A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements,
        A_scs_nnz, A_scs_chunk_ptr, A_scs_chunk_lengths, A_scs_col, A_scs_val,
        A_scs_perm);

#ifdef DEBUG_MODE
    print_vector<IT>(A_scs_perm, A_scs_n_rows);
#endif

    // Pad to the same length to emulate real iterative schemes
    IT vec_size = std::max(A_scs_n_rows_padded, A_scs_n_cols);

    DenseMatrix<VT> *x = new DenseMatrix<VT>(vec_size, 1, 1.0);
    DenseMatrix<VT> *y_smax = new DenseMatrix<VT>(vec_size, 1, 0.0);
    DenseMatrix<VT> *y_mkl = new DenseMatrix<VT>(vec_size, 1, 0.0);
    DenseMatrix<VT> *y_mkl_perm = new DenseMatrix<VT>(vec_size, 1, 0.0);

    register_kernel<IT, VT>(smax, std::string("my_scs_spmv"),
                            SMAX::KernelType::SPMV, SMAX::PlatformType::CPU);

    // A is expected to be in the SCS format
    smax->kernel("my_scs_spmv")->set_mat_scs(true);

    smax->kernel("my_scs_spmv")
        ->register_A(A_scs_C, A_scs_sigma, A_scs_n_rows, A_scs_n_rows_padded,
                     A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements, A_scs_nnz,
                     A_scs_chunk_ptr, A_scs_chunk_lengths, A_scs_col, A_scs_val,
                     A_scs_perm);
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
        y_mkl_perm->val[A_scs_perm[i]] = y_mkl->val[i];
    }

    // Compare
    compare_spmv<VT>(crs_mat->n_rows, y_smax->val, y_mkl_perm->val,
                     cli_args->matrix_file_name);

    delete x;
    delete y_smax;
    delete y_mkl;
    delete[] A_scs_chunk_ptr;
    delete[] A_scs_chunk_lengths;
    delete[] A_scs_col;
    delete[] A_scs_val;
    delete[] A_scs_perm;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_SPMV;
}