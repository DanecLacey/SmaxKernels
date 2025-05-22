#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

#include <fast_matrix_market/app/Eigen.hpp>

int main(int argc, char *argv[]) {
    //INIT_SPMM;

    int n_vectors = atoi(argv[2]);
    std::ifstream input_stream(argv[1]);

    init_pin(); // Remove pinning overhead from benchmark timing

    Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_mat;
    fast_matrix_market::read_matrix_market_eigen(input_stream, eigen_mat);
    Eigen::VectorXd eigen_x = Eigen::VectorXd::Constant(eigen_mat.cols(), 1.0);
    Eigen::VectorXd eigen_y = Eigen::VectorXd::Constant(eigen_mat.rows(), 0.0);

#if 0 
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
#endif
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

    std::function<void(bool)> lambda = [bench_name, eigen_mat, eigen_x,
                                        &eigen_y](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        eigen_y.noalias() = eigen_mat * eigen_x;
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;

    std::cout << "----------------" << std::endl;
    std::cout << "--" << bench_name << " Bench--" << std::endl;
    std::cout << argv[1] << " with " << n_threads
              << " thread(s)" << std::endl;
    std::cout << "Runtime: " << runtime << std::endl;
    std::cout << "Iterations: " << n_iter << std::endl;

    long flops_per_iter = eigen_mat.nonZeros() * n_vectors * SPMM_FLOPS_PER_NZ;
    long iter_per_second = static_cast<long>(n_iter / runtime);

    std::cout << "Performance: " << flops_per_iter * iter_per_second * F_TO_GF
              << " [GF/s]" << std::endl;
    std::cout << "----------------" << std::endl;

    // PRINT_SPMM_BENCH;
    // FINALIZE_SPMM;
    delete bench_harness;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}