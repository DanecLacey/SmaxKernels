#include "../../examples_common.hpp"
#include "../../spgemm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

// Specialize for SpGEMM
#undef MIN_NUM_ITERS
#define MIN_NUM_ITERS 10

int main(int argc, char *argv[]) {

    using IT = long long int;
    using VT = double;

    // Just to take overhead of pinning away from timers
    init_pin();

    INIT_SPGEMM(IT, VT);
    CRSMatrix<IT, VT> *crs_mat_C = new CRSMatrix<IT, VT>;

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    register_kernel<IT, VT>(smax, std::string("my_spgemm"),
                            SMAX::KernelType::SPGEMM, SMAX::PlatformType::CPU);
    if (compute_AA) {
        REGISTER_SPGEMM_DATA("my_spgemm", crs_mat_A, crs_mat_A, crs_mat_C);
    } else {
        REGISTER_SPGEMM_DATA("my_spgemm", crs_mat_A, crs_mat_B, crs_mat_C);
    }

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "smax_spgemm";
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

    // Entire (Symbolic + Numerical Phase) SpGEMM
    std::function<void(bool)> lambda = [bench_name, smax,
                                        crs_mat_C](bool warmup) {
        PARALLEL_LIKWID_MARKER_START(bench_name.c_str());
        smax->kernel("my_spgemm")->run();
        crs_mat_C->clear(); // Need to free memory allocated by SMAX
        PARALLEL_LIKWID_MARKER_STOP(bench_name.c_str());
    };

    RUN_BENCH;

    smax->utils->print_timers();

    // Just to get C_nnz //
    smax->kernel("my_spgemm")->run();
    // Just to get C_nnz //
    std::cout << "A_n_rows: " << crs_mat_A->n_rows << std::endl;
    std::cout << "A_nnz: " << crs_mat_A->nnz << std::endl;
    std::cout << "C_nnz: " << crs_mat_C->nnz << std::endl;
    PRINT_SPGEMM_BENCH(crs_mat_C->nnz);
    FINALIZE_SPGEMM;
    delete crs_mat_C;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}