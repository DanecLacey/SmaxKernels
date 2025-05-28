#include "../../examples_common.hpp"
#include "../../spgemm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPGEMM;
    CRSMatrix *crs_mat_C = new CRSMatrix();

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    smax->register_kernel("my_spgemm", SMAX::KernelType::SPGEMM);
    if (compute_AA) {
        REGISTER_SPGEMM_DATA("my_spgemm", crs_mat_A, crs_mat_A, crs_mat_C);
    } else {
        REGISTER_SPGEMM_DATA("my_spgemm", crs_mat_A, crs_mat_B, crs_mat_C);
    }

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "smax_spgemm";
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

    // Entire (Symbolic + Numerical Phase) SpGEMM
    std::function<void(bool)> lambda = [bench_name, smax,
                                        crs_mat_C](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        smax->kernel("my_spgemm")->run();
        crs_mat_C->clear(); // Need to free memory allocated by SMAX
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;

    // Just to get C_nnz //
    smax->kernel("my_spgemm")->run();
    // Just to get C_nnz //

    PRINT_SPGEMM_BENCH(crs_mat_C->nnz);
    FINALIZE_SPGEMM;
    delete bench_harness;
    delete crs_mat_C;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}