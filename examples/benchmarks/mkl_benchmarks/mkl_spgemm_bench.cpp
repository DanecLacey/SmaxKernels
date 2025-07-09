#include "../../examples_common.hpp"
#include "../../spgemm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "mkl_benchmarks_common.hpp"

// Specialize for SpGEMM
#undef MIN_NUM_ITERS
#define MIN_NUM_ITERS 10

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
    INIT_SPGEMM(IT, VT);
    sparse_matrix_t A, B, C;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

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

    // No optimization hints exist for SpGEMM
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize A");
    CHECK_MKL_STATUS(mkl_sparse_optimize(B), "mkl_sparse_optimize B");

    // Setup benchmark harness
    std::string bench_name = "mkl_spgemm";
    SETUP_BENCH(bench_name);
    std::function<void()> lambda = [bench_name, descr, A, B, &C]() {
        mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, A,
                        SPARSE_OPERATION_NON_TRANSPOSE, descr, B,
                        SPARSE_STAGE_FULL_MULT, &C);
        mkl_sparse_destroy(C); // Need to free memory allocated by MKL
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

    // Execute benchmark and print results
    RUN_BENCH;

    // Just to get C_nnz
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
    PRINT_SPGEMM_BENCH(mkl_row_start[mkl_n_rows]);

    // Clean up
    FINALIZE_SPGEMM;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy A");
    CHECK_MKL_STATUS(mkl_sparse_destroy(B), "mkl_sparse_destroy B");
    CHECK_MKL_STATUS(mkl_sparse_destroy(C), "mkl_sparse_destroy C");

    return 0;
}