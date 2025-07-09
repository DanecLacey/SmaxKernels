#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    // Set datatypes
    using IT = int;
    using VT = double;

    // Just takes pinning overhead away from timers
    init_pin();

    // Setup data structures
    INIT_SPTRSV(IT, VT);
    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *b = new DenseMatrix<VT>(crs_mat->n_cols, 1, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel name to SMAX
    std::string bench_name = "smax_sptrsv";
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPTRSV,
                            SMAX::PlatformType::CPU);
    REGISTER_SPTRSV_DATA(bench_name, crs_mat_D_plus_L, x, b);

    // Setup benchmark harness
    SETUP_BENCH(bench_name);
    std::function<void()> lambda = [smax, bench_name]() {
        smax->kernel(bench_name)->apply();
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPTRSV_BENCH;
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPTRSV;
    delete x;
    delete b;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}