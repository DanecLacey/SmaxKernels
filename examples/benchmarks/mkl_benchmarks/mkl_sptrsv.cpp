#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "mkl_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPTRSV;

    DenseMatrix *x = new DenseMatrix(crs_mat->n_cols, 1, 0.0);
    DenseMatrix *b = new DenseMatrix(crs_mat->n_cols, 1, 1.0);

    // Create MKL sparse matrix handle
    sparse_matrix_t A;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_LOWER; // Lower triangular
    descr.diag = SPARSE_DIAG_NON_UNIT;   // Non-unit diagonal

    // Create the matrix handle from CSR data
    CHECK_MKL_STATUS(mkl_sparse_d_create_csr(
                         &A, SPARSE_INDEX_BASE_ZERO, crs_mat_D_plus_L->n_rows,
                         crs_mat_D_plus_L->n_cols, crs_mat_D_plus_L->row_ptr,
                         crs_mat_D_plus_L->row_ptr + 1, crs_mat_D_plus_L->col,
                         crs_mat_D_plus_L->val),
                     "mkl_sparse_d_create_csr");

    // Optimize the matrix for SpTRSV
    CHECK_MKL_STATUS(mkl_sparse_set_sv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE,
                                            descr, MKL_AGGRESSIVE_N_OPS),
                     "mkl_sparse_set_sv_hint");
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "mkl_sptrsv";
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

    // Just to take overhead of pinning away from timers
    init_pin();

    std::function<void(bool)> lambda = [bench_name, A, descr, x,
                                        b](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)

        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, b->val,
                          x->val);

        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPTRSV_BENCH;
    FINALIZE_SPTRSV;
    delete bench_harness;
    delete x;
    delete b;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");

    return 0;
}
