#include "../../examples_common.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    // Read .mtx file and optional parameters for benchmark
    CliArgs *cli_args = new CliArgs;
    parse_cli_args(cli_args, argc, argv);

    // Read matrix with mmio
    COOMatrix *coo_mat = new COOMatrix;
    coo_mat->read_from_mtx(cli_args->matrix_file_name);

    // Convert COO to CRS to pass void pointers to SMAX
    CRSMatrix *crs_mat = new CRSMatrix;
    crs_mat->convert_coo_to_crs(coo_mat);

    // Initialize dense vectors
    DenseMatrix *x = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *y = new DenseMatrix(crs_mat->n_cols, 1, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("bench_spmv", SMAX::SPMV, SMAX::CPU);
    smax->kernels["bench_spmv"]->register_A(
        &crs_mat->n_rows, &crs_mat->n_cols, &crs_mat->nnz, &crs_mat->col,
        &crs_mat->row_ptr, &crs_mat->values);
    smax->kernels["bench_spmv"]->register_B(&crs_mat->n_cols, &x->n_cols,
                                            &x->values);
    smax->kernels["bench_spmv"]->register_C(&crs_mat->n_cols, &y->n_cols,
                                            &y->values);

    // Make lambda, and pass to the benchmarking harness
    double runtime = 0.0;
    int n_iter = MIN_NUM_ITERS;
    int n_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
#endif

    std::function<void(bool)> lambda = [smax](bool warmup) {
        smax->kernels["bench_spmv"]->run();
    };
    std::string bench_name = "smax_spmv";

    BenchHarness *bench_harness =
        new BenchHarness(bench_name, lambda, n_iter, runtime, MIN_BENCH_TIME);

    std::cout << "Running bench: " << bench_name << std::endl;

    bench_harness->warmup();
    printf("Warmup complete\n");

    bench_harness->bench();
    printf("Bench complete\n");

    PRINT_SPMV_BENCH;
    SPMV_CLEANUP;
}