#include "../../examples_common.hpp"
#include "../../spgemm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

// Specialize for SpGEMM
#undef MIN_NUM_ITERS
#define MIN_NUM_ITERS 10

// Set datatypes
#ifdef USE_MKL_ILP64
using IT = long long int;
#else
using IT = int;
#endif

using VT = double;

int main(int argc, char *argv[]) {

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPGEMM(IT, VT);
    CRSMatrix<IT, VT> *crs_mat_C = new CRSMatrix<IT, VT>;

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel name to SMAX
    std::string bench_name = "smax_spgemm";
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPGEMM,
                            SMAX::PlatformType::CPU);
    if (compute_AA) {
        REGISTER_SPGEMM_DATA(bench_name, crs_mat_A, crs_mat_A, crs_mat_C);
    } else {
        REGISTER_SPGEMM_DATA(bench_name, crs_mat_A, crs_mat_B, crs_mat_C);
    }

    // Setup benchmark harness
    SETUP_BENCH;
    INIT_LIKWID_MARKERS(bench_name);
    std::function<void()> lambda = [smax, bench_name, crs_mat_C]() {
        smax->kernel(bench_name)->apply();
        crs_mat_C->clear(); // Need to free memory allocated by SMAX
    };

    // Execute benchmark and print results
    smax->kernel(bench_name)->initialize();
    RUN_BENCH;
    smax->kernel(bench_name)->finalize();
    smax->kernel(bench_name)->run(); // Just to get C_nnz
    PRINT_SPGEMM_BENCH(crs_mat_C->nnz);
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPGEMM;
    delete crs_mat_C;
    FINALIZE_LIKWID_MARKERS;
}