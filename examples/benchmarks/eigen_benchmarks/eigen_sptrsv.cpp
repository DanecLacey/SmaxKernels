#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPTRSV;

    Eigen::VectorXd b = Eigen::VectorXd::Constant(crs_mat->n_cols, 1.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(crs_mat->n_rows);

    // Convert to Eigen SparseMatrix
    Eigen::SparseMatrix<double> eigen_mat(crs_mat->n_rows, crs_mat->n_cols);
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

    std::string bench_name = "eigen_sptrsv";
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

    init_pin();

    std::function<void(bool)> lambda = [bench_name, &eigen_mat, &b,
                                        &x](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        x.noalias() = eigen_mat.triangularView<Eigen::Lower>().solve(b);
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPTRSV_BENCH;
    FINALIZE_SPTRSV;

    delete bench_harness;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}