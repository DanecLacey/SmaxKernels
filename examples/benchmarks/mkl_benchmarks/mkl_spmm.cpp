#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "mkl_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPMM;

    DenseMatrix *X = new DenseMatrix(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix *Y = new DenseMatrix(crs_mat->n_cols, n_vectors, 0.0);

    // Create MKL sparse matrix handle
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Create the matrix handle from CSR data
    CHECK_MKL_STATUS(mkl_sparse_d_create_csr(
                         &A, SPARSE_INDEX_BASE_ZERO, crs_mat->n_rows,
                         crs_mat->n_cols, crs_mat->row_ptr,
                         crs_mat->row_ptr + 1, crs_mat->col, crs_mat->val),
                     "mkl_sparse_d_create_csr");

    // Optimize the matrix for SpMV
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "mkl_spmm";
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

    std::function<void(bool)> lambda = [bench_name, A, descr, X, crs_mat,
                                        n_vectors, Y](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)

        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        1.0, // alpha
                        A, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X->val, n_vectors,
                        crs_mat->n_cols, // leading dimension of X
                        0.0,             // beta
                        Y->val,
                        crs_mat->n_rows // leading dimension of Y
        );

        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPMM_BENCH;
    FINALIZE_SPMM;
    delete bench_harness;
    delete X;
    delete Y;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");

    return 0;
}