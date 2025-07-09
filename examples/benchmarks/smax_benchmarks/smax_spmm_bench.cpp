#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    // Set datatypes
    using IT = int;
    using VT = double;

    // Just takes pinning overhead away from timers
    init_pin();

    // Setup data structures
    INIT_SPMM(IT, VT);
    DenseMatrix<VT> *X = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix<VT> *Y = new DenseMatrix<VT>(crs_mat->n_cols, n_vectors, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel name to SMAX
    std::string bench_name = "smax_spmm";
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPMM,
                            SMAX::PlatformType::CPU);
    REGISTER_SPMM_DATA(bench_name, crs_mat, n_vectors, X, Y);

    // Setup benchmark harness
    SETUP_BENCH(bench_name);
    std::function<void()> lambda = [smax, bench_name]() {
        smax->kernel(bench_name)->apply();
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMM_BENCH;
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPMM;
    delete X;
    delete Y;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}