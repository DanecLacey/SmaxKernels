#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    // Set datatypes
    using IT = int;
    using VT = double;

    // Just takes pinning overhead away from timers
    init_pin();

    // Setup data structures
    INIT_SPMM(IT, VT);

    Eigen::MatrixXd eigen_x =
        Eigen::MatrixXd::Constant(crs_mat->n_cols, n_vectors, 1.0);
    Eigen::MatrixXd eigen_y =
        Eigen::MatrixXd::Constant(crs_mat->n_rows, n_vectors, 0.0);

    // Wrap your CRS data into an Eigen SparseMatrix
    Eigen::SparseMatrix<VT, Eigen::RowMajor> eigen_mat(crs_mat->n_rows,
                                                       crs_mat->n_cols);
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
    std::string bench_name = "eigen_spmm";
    SETUP_BENCH(bench_name);

    std::function<void()> lambda = [bench_name, eigen_mat, eigen_x,
                                    &eigen_y]() {
        eigen_y.noalias() = eigen_mat * eigen_x;
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMM_BENCH;

    // Clean up
    FINALIZE_SPMM;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}