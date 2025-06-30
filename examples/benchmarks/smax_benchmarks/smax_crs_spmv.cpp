#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    using IT = unsigned int;
    using VT = double;

    // Just to take overhead of pinning away from timers
    init_pin();

    INIT_SPMV(IT, VT);

    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *y = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    register_kernel<IT, VT>(smax, std::string("my_crs_spmv"),
                            SMAX::KernelType::SPMV, SMAX::PlatformType::CPU);
    REGISTER_SPMV_DATA("my_crs_spmv", crs_mat, x, y);

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "smax_crs_spmv";
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

    std::function<void(bool)> lambda = [bench_name, smax](bool warmup) {
        PARALLEL_LIKWID_MARKER_START(bench_name.c_str());
        smax->kernel("my_crs_spmv")->run();
        PARALLEL_LIKWID_MARKER_STOP(bench_name.c_str());
    };

    RUN_BENCH;
    PRINT_SPMV_BENCH;

    smax->utils->print_timers();

    FINALIZE_SPMV;
    delete bench_harness;
    delete x;
    delete y;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}