#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

// Set datatypes
using IT = unsigned int;
using VT = double;

int main(int argc, char *argv[]) {

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPMV(IT, VT);
    IT _C = cli_args->_C;
    IT _sigma = cli_args->_sigma;
    SCSMatrix<IT, VT> *scs_mat = new SCSMatrix<IT, VT>(_C, _sigma);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Convert CRS to Sell-C-sigma matrix
    smax->utils->convert_crs_to_scs<IT, VT, ULL>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, scs_mat->C, scs_mat->sigma,
        scs_mat->n_rows, scs_mat->n_rows_padded, scs_mat->n_cols,
        scs_mat->n_chunks, scs_mat->n_elements, scs_mat->nnz,
        scs_mat->chunk_ptr, scs_mat->chunk_lengths, scs_mat->col, scs_mat->val,
        scs_mat->perm);

    // Pad to the same length to emulate real iterative schemes
    IT vec_size = std::max(scs_mat->n_rows_padded, scs_mat->n_cols);
    DenseMatrix<VT> *x = new DenseMatrix<VT>(vec_size, 1, 1.0);
    DenseMatrix<VT> *y = new DenseMatrix<VT>(vec_size, 1, 0.0);

    // Register kernel name to SMAX
    std::ostringstream oss;
    oss << "smax_SELL_" << _C << "_" << _sigma << "_spmv";
    std::string bench_name = oss.str();
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPMV,
                            SMAX::PlatformType::CPU);

    // Tell SMAX that A is expected to be in the SCS format
    smax->kernel(bench_name)->set_mat_scs(true);

    // Register kenel data
    smax->kernel(bench_name)
        ->register_A(scs_mat->C, scs_mat->sigma, scs_mat->n_rows,
                     scs_mat->n_rows_padded, scs_mat->n_cols, scs_mat->n_chunks,
                     scs_mat->n_elements, scs_mat->nnz, scs_mat->chunk_ptr,
                     scs_mat->chunk_lengths, scs_mat->col, scs_mat->val,
                     scs_mat->perm);
    smax->kernel(bench_name)->register_B(vec_size, x->val);
    smax->kernel(bench_name)->register_C(vec_size, y->val);

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
    PRINT_SPMV_BENCH;
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPMV;
    delete scs_mat;
    delete x;
    delete y;
    FINALIZE_LIKWID_MARKERS;
}