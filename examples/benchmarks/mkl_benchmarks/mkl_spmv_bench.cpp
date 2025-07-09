#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
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

    // Setup benchmark harness
    std::string bench_name = "mkl_spmv";
    SETUP_BENCH(bench_name);

    std::function<void()> lambda = [bench_name, A, descr, x, y]() {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x->val,
                        0.0, y->val);
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMV_BENCH;

    // Clean up
    FINALIZE_SPMV;
    delete x;
    delete y;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");

    return 0;
}