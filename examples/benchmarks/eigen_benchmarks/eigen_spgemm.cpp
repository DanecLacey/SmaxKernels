#include "../../examples_common.hpp"
#include "../../spgemm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    using IT = int;
    using VT = double;

    init_pin(); // avoid counting pinning in timing

    INIT_SPGEMM(IT, VT);
    CRSMatrix<IT, VT> *crs_mat_C = new CRSMatrix<IT, VT>();

    Eigen::SparseMatrix<VT> eigen_A(crs_mat_A->n_rows, crs_mat_A->n_cols);
    Eigen::SparseMatrix<VT> eigen_B(crs_mat_B->n_rows, crs_mat_B->n_cols);
    {
        std::vector<Eigen::Triplet<VT>> triA;
        triA.reserve(crs_mat_A->nnz);
        for (IT i = 0; i < crs_mat_A->n_rows; ++i)
            for (IT k = crs_mat_A->row_ptr[i]; k < crs_mat_A->row_ptr[i + 1];
                 ++k)
                triA.emplace_back(i, crs_mat_A->col[k], crs_mat_A->val[k]);
        eigen_A.setFromTriplets(triA.begin(), triA.end());
    }
    {
        std::vector<Eigen::Triplet<VT>> triB;
        triB.reserve(crs_mat_B->nnz);
        for (IT i = 0; i < crs_mat_B->n_rows; ++i)
            for (IT k = crs_mat_B->row_ptr[i]; k < crs_mat_B->row_ptr[i + 1];
                 ++k)
                triB.emplace_back(i, crs_mat_B->col[k], crs_mat_B->val[k]);
        eigen_B.setFromTriplets(triB.begin(), triB.end());
    }

    Eigen::SparseMatrix<VT> eigen_C(crs_mat_A->n_rows, crs_mat_B->n_cols);

    // --- Benchmark metadata ---
    std::string bench_name = "eigen_spgemm";
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

    std::function<void(bool)> lambda = [bench_name, &eigen_A, &eigen_B,
                                        &eigen_C](bool warmup) {
        PARALLEL_LIKWID_MARKER_START(bench_name.c_str());
        eigen_C = eigen_A * eigen_B;
        PARALLEL_LIKWID_MARKER_STOP(bench_name.c_str());
    };

    RUN_BENCH;
    PRINT_SPGEMM_BENCH(eigen_C.nonZeros());
    FINALIZE_SPGEMM;
    delete bench_harness;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}