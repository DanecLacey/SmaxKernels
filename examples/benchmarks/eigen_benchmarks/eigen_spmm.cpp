#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPMM;

    Eigen::MatrixXd eigen_x =
        Eigen::MatrixXd::Constant(crs_mat->n_cols, n_vectors, 1.0);
    Eigen::MatrixXd eigen_y =
        Eigen::MatrixXd::Constant(crs_mat->n_rows, n_vectors, 0.0);

    // Wrap your CRS data into an Eigen SparseMatrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_mat(crs_mat->n_rows, crs_mat->n_cols);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(crs_mat->nnz);

    for (int i = 0; i < crs_mat->n_rows; ++i) {
        for (int idx = crs_mat->row_ptr[i]; idx < crs_mat->row_ptr[i + 1];
             ++idx) {
            int j = crs_mat->col[idx];
            double val = crs_mat->val[idx];
            triplets.emplace_back(i, j, val);
        }
    }
    eigen_mat.setFromTriplets(triplets.begin(), triplets.end());

    // Setup benchmark metadata
    std::string bench_name = "eigen_spmm";
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

    init_pin(); // Remove pinning overhead from benchmark timing

    std::function<void(bool)> lambda = [bench_name, eigen_mat, eigen_x,
                                        &eigen_y](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        eigen_y.noalias() = eigen_mat * eigen_x;
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPMM_BENCH;
    FINALIZE_SPMM;
    delete bench_harness;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}