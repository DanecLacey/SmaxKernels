#include "../../examples_common.hpp"
#include "../../sptrsm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "mkl_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_MKL_ILP64
    using IT = long long int;
#else
    using IT = int;
#endif
    using VT = double;

    // Just to take overhead of pinning away from timers
    init_pin();

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

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "mkl_sptrsm";
    double runtime = 0.0;
    int n_iter = MIN_NUM_ITERS;
    int n_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
#endif
#ifdef USE_LIKWID
    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
        LIKWID_MARKER_REGISTER(bench_name.c_str());
    }
#endif

    std::function<void(bool)> lambda = [bench_name, A, descr, X, crs_mat,
                                        n_vectors, B](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr,
                          SPARSE_LAYOUT_COLUMN_MAJOR, B->val, n_vectors,
                          crs_mat->n_rows, X->val, crs_mat->n_rows);
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPTRSM_BENCH;
    FINALIZE_SPTRSM;
    delete bench_harness;
    delete X;
    delete B;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");

    return 0;
}