#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
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

    INIT_SPMV(IT, VT);

    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *y = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

    // Create MKL sparse matrix handle
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

    // Optimize the matrix for SpMV
    CHECK_MKL_STATUS(mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE,
                                            descr, MKL_AGGRESSIVE_N_OPS),
                     "mkl_sparse_set_mv_hint");
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "mkl_spmv";
    float runtime = 0.0;
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

    std::function<void(bool)> lambda = [bench_name, A, descr, x,
                                        y](bool warmup) {
        PARALLEL_LIKWID_MARKER_START(bench_name.c_str());
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x->val,
                        0.0, y->val);
        PARALLEL_LIKWID_MARKER_STOP(bench_name.c_str());
    };

    RUN_BENCH;
    PRINT_SPMV_BENCH;
    FINALIZE_SPMV;
    delete bench_harness;
    delete x;
    delete y;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");

    return 0;
}