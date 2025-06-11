#include "../../examples_common.hpp"
#include "../../spgemm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "mkl_benchmarks_common.hpp"

// Specialize for SpGEMM
#undef MIN_NUM_ITERS
#define MIN_NUM_ITERS 10

int main(int argc, char *argv[]) {
    // Just to take overhead of pinning away from timers
    init_pin();

    INIT_SPGEMM;

    // Create MKL sparse matrix handles for matrices A and B
    sparse_matrix_t A, B, C;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    CHECK_MKL_STATUS(
        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, crs_mat_A->n_rows,
                                crs_mat_A->n_cols, crs_mat_A->row_ptr,
                                crs_mat_A->row_ptr + 1, crs_mat_A->col,
                                crs_mat_A->val),
        "mkl_sparse_d_create_csr A.")

    if (compute_AA) {
        CHECK_MKL_STATUS(
            mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO,
                                    crs_mat_A->n_rows, crs_mat_A->n_cols,
                                    crs_mat_A->row_ptr, crs_mat_A->row_ptr + 1,
                                    crs_mat_A->col, crs_mat_A->val),
            "mkl_sparse_d_create_csr B.")
    } else {
        CHECK_MKL_STATUS(
            mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO,
                                    crs_mat_B->n_rows, crs_mat_B->n_cols,
                                    crs_mat_B->row_ptr, crs_mat_B->row_ptr + 1,
                                    crs_mat_B->col, crs_mat_B->val),
            "mkl_sparse_d_create_csr B.")
    }

    // No optimization hints exist for SpGEMM
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize A");
    CHECK_MKL_STATUS(mkl_sparse_optimize(B), "mkl_sparse_optimize B");

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "mkl_spgemm";
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

    // Entire (Symbolic + Numerical Phase) SpGEMM
    std::function<void(bool)> lambda = [bench_name, descr, A, B,
                                        &C](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, A,
                        SPARSE_OPERATION_NON_TRANSPOSE, descr, B,
                        SPARSE_STAGE_FULL_MULT, &C);
        mkl_sparse_destroy(C); // Need to free memory allocated by MKL
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    // TODO: Include option more cleanly
    // SpGEMM Symbolic Phase
    // std::function<void(bool)> lambda = [bench_name, A, B, &C](bool warmup) {
    //     IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
    //     mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, A,
    //                     SPARSE_OPERATION_NON_TRANSPOSE, descr, B,
    //                     SPARSE_STAGE_FULL_MULT_NO_VAL, &C);
    //     mkl_sparse_destroy(C); // Need to free memory allocated by MKL
    //     IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    // };

    RUN_BENCH;

    // Just to get C_nnz //
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, B, &C);
    MKL_INT mkl_n_rows;
    MKL_INT mkl_n_cols;
    MKL_INT *mkl_row_start = nullptr;
    MKL_INT *mkl_row_end = nullptr;
    MKL_INT *mkl_col = nullptr;
    double *mkl_val = nullptr;
    sparse_index_base_t indexing;
    CHECK_MKL_STATUS(mkl_sparse_d_export_csr(C, &indexing, &mkl_n_rows,
                                             &mkl_n_cols, &mkl_row_start,
                                             &mkl_row_end, &mkl_col, &mkl_val),
                     "Failed to export matrix C.")
    // Just to get C_nnz //
    std::cout << "A_n_rows: " << crs_mat_A->n_rows << std::endl;
    std::cout << "A_nnz: " << crs_mat_A->nnz << std::endl;
    std::cout << "C_nnz: " << mkl_row_start[mkl_n_rows] << std::endl;
    PRINT_SPGEMM_BENCH(mkl_row_start[mkl_n_rows]);
    FINALIZE_SPGEMM;
    delete bench_harness;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy A");
    CHECK_MKL_STATUS(mkl_sparse_destroy(B), "mkl_sparse_destroy B");
    CHECK_MKL_STATUS(mkl_sparse_destroy(C), "mkl_sparse_destroy C");

    return 0;
}