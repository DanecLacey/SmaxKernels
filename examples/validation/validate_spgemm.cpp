#include "../examples_common.hpp"
#include "../spgemm_helpers.hpp"
#include "validation_common.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    INIT_SPGEMM(IT, VT);
    CRSMatrix<IT, VT> *crs_mat_C_smax = new CRSMatrix<IT, VT>;
    CRSMatrix<IT, VT> *crs_mat_C_mkl = new CRSMatrix<IT, VT>;

    // Smax SpGEMM
    SMAX::Interface *smax = new SMAX::Interface();
    register_kernel<IT, VT>(smax, std::string("my_spgemm"),
                            SMAX::KernelType::SPGEMM, SMAX::PlatformType::CPU);

    if (compute_AA) {
        REGISTER_SPGEMM_DATA("my_spgemm", crs_mat_A, crs_mat_A, crs_mat_C_smax);
    } else {
        REGISTER_SPGEMM_DATA("my_spgemm", crs_mat_A, crs_mat_B, crs_mat_C_smax);
    }

    smax->kernel("my_spgemm")->run();

    // MKL SpGEMM
    sparse_matrix_t A, B, C;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Create MKL sparse matrices from CSR data
    CHECK_MKL_STATUS(
        mkl_sparse_d_create_csr(
            /* handle    */ &A,
            /* indexing  */ SPARSE_INDEX_BASE_ZERO,
            /* rows      */ static_cast<MKL_INT>(crs_mat_A->n_rows),
            /* cols      */ static_cast<MKL_INT>(crs_mat_A->n_cols),
            /* row_start */ reinterpret_cast<MKL_INT *>(crs_mat_A->row_ptr),
            /* row_end   */ reinterpret_cast<MKL_INT *>(crs_mat_A->row_ptr + 1),
            /* col_ind   */ reinterpret_cast<MKL_INT *>(crs_mat_A->col),
            /* values    */ crs_mat_A->val),
        "mkl_sparse_d_create_csr");

    if (compute_AA) {
        CHECK_MKL_STATUS(
            mkl_sparse_d_create_csr(
                /* handle    */ &B,
                /* indexing  */ SPARSE_INDEX_BASE_ZERO,
                /* rows      */ static_cast<MKL_INT>(crs_mat_A->n_rows),
                /* cols      */ static_cast<MKL_INT>(crs_mat_A->n_cols),
                /* row_start */ reinterpret_cast<MKL_INT *>(crs_mat_A->row_ptr),
                /* row_end   */
                reinterpret_cast<MKL_INT *>(crs_mat_A->row_ptr + 1),
                /* col_ind   */ reinterpret_cast<MKL_INT *>(crs_mat_A->col),
                /* values    */ crs_mat_A->val),
            "mkl_sparse_d_create_csr");
    } else {
        CHECK_MKL_STATUS(
            mkl_sparse_d_create_csr(
                /* handle    */ &B,
                /* indexing  */ SPARSE_INDEX_BASE_ZERO,
                /* rows      */ static_cast<MKL_INT>(crs_mat_B->n_rows),
                /* cols      */ static_cast<MKL_INT>(crs_mat_B->n_cols),
                /* row_start */ reinterpret_cast<MKL_INT *>(crs_mat_B->row_ptr),
                /* row_end   */
                reinterpret_cast<MKL_INT *>(crs_mat_B->row_ptr + 1),
                /* col_ind   */ reinterpret_cast<MKL_INT *>(crs_mat_B->col),
                /* values    */ crs_mat_B->val),
            "mkl_sparse_d_create_csr");
    }

    // Entire (Symbolic + Numerical Phase) SpGEMM
    CHECK_MKL_STATUS(mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, A,
                                     SPARSE_OPERATION_NON_TRANSPOSE, descr, B,
                                     SPARSE_STAGE_FULL_MULT, &C),
                     "mkl_sparse_sp2m");

    // Export the result matrix C in CSR format
    sparse_index_base_t indexing;
    MKL_INT mkl_n_rows;
    MKL_INT mkl_n_cols;
    MKL_INT *mkl_row_start = nullptr;
    MKL_INT *mkl_row_end = nullptr;
    MKL_INT *mkl_col = nullptr;
    VT *mkl_val = nullptr;
    CHECK_MKL_STATUS(mkl_sparse_d_export_csr(C, &indexing, &mkl_n_rows,
                                             &mkl_n_cols, &mkl_row_start,
                                             &mkl_row_end, &mkl_col, &mkl_val),
                     "Failed to export matrix C.")

    // Must do this manually
    crs_mat_C_mkl->n_rows = mkl_n_rows;
    crs_mat_C_mkl->n_cols = mkl_n_cols;
    crs_mat_C_mkl->nnz =
        mkl_row_end[crs_mat_C_mkl->n_rows - 1] - mkl_row_start[0];
    crs_mat_C_mkl->row_ptr = new MKL_INT[crs_mat_C_mkl->n_rows + 1];
    crs_mat_C_mkl->col = new MKL_INT[crs_mat_C_mkl->nnz];
    crs_mat_C_mkl->val = new VT[crs_mat_C_mkl->nnz];

    for (ULL i = 0; i <= crs_mat_C_mkl->n_rows; ++i) {
        crs_mat_C_mkl->row_ptr[i] = mkl_row_start[i]; // ?
    }
    for (ULL i = 0; i < crs_mat_C_mkl->nnz; ++i) {
        crs_mat_C_mkl->col[i] = mkl_col[i];
        crs_mat_C_mkl->val[i] = mkl_val[i];
    }

    // printf("MKL: \n");
    // crs_mat_C_mkl->print();
    // printf("SMAX: \n");
    // crs_mat_C_smax->print();
    smax->utils->print_timers();

    std::cout << "MKL nnz = " << crs_mat_C_mkl->nnz << std::endl;
    std::cout << "SMAX nnz = " << crs_mat_C_smax->nnz << std::endl;

    // Compare
    compare_spgemm<IT, VT>(crs_mat_C_smax, crs_mat_C_mkl,
                           cli_args->matrix_file_name_A,
                           cli_args->matrix_file_name_B);

    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    CHECK_MKL_STATUS(mkl_sparse_destroy(B), "mkl_sparse_destroy");
    CHECK_MKL_STATUS(mkl_sparse_destroy(C), "mkl_sparse_destroy");
    delete crs_mat_C_smax;
    delete crs_mat_C_mkl;
    FINALIZE_SPGEMM;
}
