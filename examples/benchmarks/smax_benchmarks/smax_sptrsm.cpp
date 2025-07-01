#include "../../examples_common.hpp"
#include "../../sptrsm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    using IT = int;
    using VT = double;

    // Just to take overhead of pinning away from timers
    init_pin();

    INIT_SPTRSM(IT, VT);
    DenseMatrix<VT> *X = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix<VT> *B = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    register_kernel<IT, VT>(smax, std::string("my_sptrsm"),
                            SMAX::KernelType::SPTRSM, SMAX::PlatformType::CPU);
    REGISTER_SPTRSM_DATA("my_sptrsm", crs_mat_D_plus_L, X, B);

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "smax_sptrsm";
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
        smax->kernel("my_sptrsm")->run();
        PARALLEL_LIKWID_MARKER_STOP(bench_name.c_str());
    };

    RUN_BENCH;
    PRINT_SPTRSM_BENCH;
    FINALIZE_SPTRSM;
    delete bench_harness;
    delete X;
    delete B;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}