#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "mkl_benchmarks_common.hpp"

// Set datatypes
#ifdef USE_MKL_ILP64
using IT = long long int;
#else
using IT = int;
#endif
using VT = double;

int main(int argc, char *argv[]) {

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPTRSV(IT, VT);
    CRSMatrix<IT, VT> *crs_mat_D_plus_L = new CRSMatrix<IT, VT>;
    CRSMatrix<IT, VT> *crs_mat_U = new CRSMatrix<IT, VT>;
    extract_D_L_U<IT, VT>(*crs_mat, *crs_mat_D_plus_L, *crs_mat_U);
    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 0.0);
    DenseMatrix<VT> *b = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);

    // Create MKL sparse matrix handle
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

    // Optimize the matrix for SpTRSV
    CHECK_MKL_STATUS(mkl_sparse_set_sv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE,
                                            descr, MKL_AGGRESSIVE_N_OPS),
                     "mkl_sparse_set_sv_hint");
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "mkl_sptrsv";
    SETUP_BENCH;
    INIT_LIKWID_MARKERS(bench_name);

    std::function<void()> lambda = [bench_name, A, descr, x, b]() {
        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, b->val,
                          x->val);
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPTRSV_BENCH;

    // Clean up
    FINALIZE_SPTRSV;
    delete x;
    delete b;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_LIKWID_MARKERS;
}
