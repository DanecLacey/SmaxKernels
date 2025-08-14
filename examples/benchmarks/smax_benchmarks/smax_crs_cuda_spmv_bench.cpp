#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

// Set datatypes
using IT = int;
using VT = float;

int main(int argc, char *argv[]) {

    // Setup data structures
    INIT_SPMV(IT, VT);

    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *y = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel name to SMAX
    std::string bench_name = "smax_crs_cuda_spmv";
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPMV,
                            SMAX::PlatformType::CUDA);

    // Register kenel data
    REGISTER_SPMV_DATA(bench_name, crs_mat, x, y);

    // Setup benchmark harness
    SETUP_BENCH;
    std::function<void()> lambda = [smax, bench_name]() {
        smax->kernel(bench_name)->apply();
    };

    // Execute benchmark and print results
    smax->kernel(bench_name)->initialize();
    RUN_BENCH;
    smax->kernel(bench_name)->finalize();
    PRINT_SPMV_BENCH;
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPMV;
    delete x;
    delete y;
    delete smax;
}