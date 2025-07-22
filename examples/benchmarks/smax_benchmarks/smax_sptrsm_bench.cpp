#include "../../examples_common.hpp"
#include "../../sptrsm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

// Set datatypes
using IT = int;
using VT = double;

int main(int argc, char *argv[]) {

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPTRSM(IT, VT);
    DenseMatrix<VT> *X = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix<VT> *B = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel name to SMAX
    std::string bench_name = "smax_sptrsm";
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPTRSM,
                            SMAX::PlatformType::CPU);
    smax->kernel(bench_name)->set_vec_row_major(true);
    REGISTER_SPTRSM_DATA(bench_name, crs_mat_D_plus_L, X, B);

    // Setup benchmark harness
    SETUP_BENCH;
    INIT_LIKWID_MARKERS(bench_name);
    std::function<void()> lambda = [smax, bench_name]() {
        smax->kernel(bench_name)->apply();
    };

    // Execute benchmark and print results
    smax->kernel(bench_name)->initialize();
    RUN_BENCH;
    smax->kernel(bench_name)->finalize();
    PRINT_SPTRSM_BENCH;
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPTRSM;
    delete X;
    delete B;
    FINALIZE_LIKWID_MARKERS;
}