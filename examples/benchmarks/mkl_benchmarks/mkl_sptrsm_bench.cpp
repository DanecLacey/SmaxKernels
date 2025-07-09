#include "../../examples_common.hpp"
#include "../../sptrsm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "mkl_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    // Set datatypes
#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    // Just takes pinning overhead away from timers
    init_pin();

    // Setup data structures
    INIT_SPTRSM(IT, VT);
    DenseMatrix<VT> *X = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix<VT> *B = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 0.0);

    // Create MKL sparse matrix handle
    sparse_matrix_t A;
    matrix_descr descr;
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

    // Optimize the matrix for SpTRSM
    // Assume row-major dense vectors by default
    // TODO: Should not fail
    // CHECK_MKL_STATUS(mkl_sparse_set_sm_hint(A,
    // SPARSE_OPERATION_NON_TRANSPOSE,
    //                                         descr, SPARSE_LAYOUT_ROW_MAJOR,
    //                                         n_vectors, MKL_AGGRESSIVE_N_OPS),
    //                  "mkl_sparse_set_sm_hint");
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    // Setup benchmark harness
    std::string bench_name = "mkl_sptrsm";
    SETUP_BENCH(bench_name);

    std::function<void()> lambda = [bench_name, A, descr, X, crs_mat, n_vectors,
                                    B]() {
        mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr,
                          SPARSE_LAYOUT_COLUMN_MAJOR, B->val, n_vectors,
                          crs_mat->n_rows, X->val, crs_mat->n_rows);
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPTRSM_BENCH;

    // Clean up
    FINALIZE_SPTRSM;
    delete X;
    delete B;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");

    return 0;
}