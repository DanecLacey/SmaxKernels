#include "../../examples_common.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

int main(int argc, char *argv[]) {
    INIT_MTX;

    Eigen::VectorXd eigen_x = Eigen::VectorXd::Constant(crs_mat->n_cols, 1.0);
    Eigen::VectorXd eigen_y = Eigen::VectorXd::Constant(crs_mat->n_rows, 0.0);

    // Wrap your CRS data into an Eigen SparseMatrix
    Eigen::SparseMatrix<double> eigen_mat(crs_mat->n_rows, crs_mat->n_cols);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(crs_mat->nnz);

    for (int i = 0; i < crs_mat->n_rows; ++i) {
        for (int idx = crs_mat->row_ptr[i]; idx < crs_mat->row_ptr[i + 1];
             ++idx) {
            int j = crs_mat->col[idx];
            double val = crs_mat->values[idx];
            triplets.emplace_back(i, j, val);
        }
    }
    eigen_mat.setFromTriplets(triplets.begin(), triplets.end());

    // Make lambda, and pass to the benchmarking harness
    double runtime = 0.0;
    int n_iter = MIN_NUM_ITERS;
    int n_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
#endif

    std::function<void(bool)> lambda = [eigen_mat, eigen_x,
                                        &eigen_y](bool warmup) {
        eigen_y = eigen_mat * eigen_x;
    };

    std::string bench_name = "eigen_spmv";

    RUN_BENCH;
    PRINT_SPMV_BENCH;
    SPMV_CLEANUP;
}