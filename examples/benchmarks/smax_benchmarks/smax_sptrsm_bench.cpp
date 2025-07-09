#include "../../examples_common.hpp"
#include "../../sptrsm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    // Set datatypes
    using IT = int;
    using VT = double;

    // Just takes pinning overhead away from timers
    init_pin();

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
    REGISTER_SPTRSM_DATA(bench_name, crs_mat_D_plus_L, X, B);

    // Setup benchmark harness
    SETUP_BENCH(bench_name);
    std::function<void()> lambda = [smax, bench_name]() {
        smax->kernel(bench_name)->apply();
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPTRSM_BENCH;
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPTRSM;
    delete X;
    delete B;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}