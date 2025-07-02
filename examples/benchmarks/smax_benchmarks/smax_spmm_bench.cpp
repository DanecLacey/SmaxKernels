#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    using IT = int;
    using VT = double;

    // Just to take overhead of pinning away from timers
    init_pin();

    INIT_SPMM(IT, VT);
    DenseMatrix<VT> *X = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix<VT> *Y = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    register_kernel<IT, VT>(smax, std::string("my_spmm"),
                            SMAX::KernelType::SPMM, SMAX::PlatformType::CPU);

    REGISTER_SPMM_DATA("my_spmm", crs_mat, n_vectors, X, Y);

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "smax_spmm";
    float runtime = 0.0;
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
        smax->kernel("my_spmm")->run();
        PARALLEL_LIKWID_MARKER_STOP(bench_name.c_str());
    };

    RUN_BENCH;
    PRINT_SPMM_BENCH;
    FINALIZE_SPMM;
    delete X;
    delete Y;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}