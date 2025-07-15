#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
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
    INIT_SPMM(IT, VT);
    DenseMatrix<VT> *X = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix<VT> *Y = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 0.0);

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

    // Optimize the matrix for SpMM
    // Assume row-major dense vectors by default
    CHECK_MKL_STATUS(mkl_sparse_set_mm_hint(A, SPARSE_OPERATION_NON_TRANSPOSE,
                                            descr, SPARSE_LAYOUT_ROW_MAJOR,
                                            n_vectors, MKL_AGGRESSIVE_N_OPS),
                     "mkl_sparse_set_mm_hint");
    CHECK_MKL_STATUS(mkl_sparse_optimize(A), "mkl_sparse_optimize");

    // Setup benchmark harness
    std::string bench_name = "mkl_spmm";
    SETUP_BENCH;
    INIT_LIKWID_MARKERS(bench_name);

    std::function<void()> lambda = [bench_name, A, descr, X, crs_mat, n_vectors,
                                    Y]() {
        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        1.0, // alpha
                        A, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X->val, n_vectors,
                        crs_mat->n_cols, // leading dimension of X
                        0.0,             // beta
                        Y->val,
                        crs_mat->n_rows // leading dimension of Y
        );
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMM_BENCH;

    // Clean up
    FINALIZE_SPMM;

    delete X;
    delete Y;
    CHECK_MKL_STATUS(mkl_sparse_destroy(A), "mkl_sparse_destroy");
    FINALIZE_LIKWID_MARKERS;
}