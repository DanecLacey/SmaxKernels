#include "../../examples_common.hpp"
#include "../../sptrsm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

// Set datatypes
using IT = int;
using VT = double;

int main(int argc, char *argv[]) {

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPTRSM(IT, VT);

    Eigen::MatrixXd X =
        Eigen::MatrixXd::Constant(crs_mat->n_cols, n_vectors, 1.0);
    Eigen::MatrixXd B =
        Eigen::MatrixXd::Constant(crs_mat->n_rows, n_vectors, 0.0);

    // Wrap your CRS data into an Eigen SparseMatrix
    Eigen::SparseMatrix<VT> eigen_mat(crs_mat->n_rows, crs_mat->n_cols);
    std::vector<Eigen::Triplet<VT>> triplets;
    triplets.reserve(crs_mat->nnz);

    for (IT i = 0; i < crs_mat->n_rows; ++i) {
        for (IT idx = crs_mat->row_ptr[i]; idx < crs_mat->row_ptr[i + 1];
             ++idx) {
            IT j = crs_mat->col[idx];
            VT val = crs_mat->val[idx];
            triplets.emplace_back(i, j, val);
        }
    }
    eigen_mat.setFromTriplets(triplets.begin(), triplets.end());

    // Setup benchmark harness
    std::string bench_name = "eigen_sptrsm";
    SETUP_BENCH;
    INIT_LIKWID_MARKERS(bench_name);

    std::function<void()> lambda = [bench_name, &eigen_mat, &X, &B]() {
        X.noalias() = eigen_mat.triangularView<Eigen::Lower>().solve(B);
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPTRSM_BENCH;

    // Clean up
    FINALIZE_SPTRSM;
    FINALIZE_LIKWID_MARKERS;
}