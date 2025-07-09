#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    // Set datatypes
    using IT = int;
    using VT = double;

    // Just takes pinning overhead away from timers
    init_pin();

    // Setup data structures
    INIT_SPTRSV(IT, VT);

    Eigen::VectorXd b = Eigen::VectorXd::Constant(crs_mat->n_cols, 1.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(crs_mat->n_rows);

    // Convert to Eigen SparseMatrix
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
    std::string bench_name = "eigen_sptrsv";
    SETUP_BENCH(bench_name);

    std::function<void()> lambda = [bench_name, &eigen_mat, &x, &b]() {
        x.noalias() = eigen_mat.triangularView<Eigen::Lower>().solve(b);
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPTRSV_BENCH;

    // Clean up
    FINALIZE_SPTRSV;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}