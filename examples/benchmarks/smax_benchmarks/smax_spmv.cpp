#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPMV;
    DenseMatrix *x = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *y = new DenseMatrix(crs_mat->n_cols, 1, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    smax->register_kernel("my_spmv", SMAX::SPMV, SMAX::CPU);
    REGISTER_SPMV_DATA("my_spmv", crs_mat, x, y);

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "smax_spmv";
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

    // Just to take overhead of pinning away from timers
    init_pin();

    std::function<void(bool)> lambda = [bench_name, smax](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        smax->kernels["my_spmv"]->run();
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPMV_BENCH;
    FINALIZE_SPMV;
    delete bench_harness;
    delete x;
    delete y;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}