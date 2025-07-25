#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

// Set datatypes
using IT = int;
using VT = double;

int main(int argc, char *argv[]) {

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPTRSV(IT, VT);
    CRSMatrix<IT, VT> *crs_mat_D_plus_L = new CRSMatrix<IT, VT>;
    CRSMatrix<IT, VT> *crs_mat_U = new CRSMatrix<IT, VT>;
    extract_D_L_U<IT, VT>(*crs_mat, *crs_mat_D_plus_L, *crs_mat_U);

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
    SETUP_BENCH;
    INIT_LIKWID_MARKERS(bench_name);

    std::function<void()> lambda = [bench_name, &eigen_mat, &x, &b]() {
        x.noalias() = eigen_mat.triangularView<Eigen::Lower>().solve(b);
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPTRSV_BENCH;

    // Clean up
    FINALIZE_SPTRSV;
    FINALIZE_LIKWID_MARKERS;
}