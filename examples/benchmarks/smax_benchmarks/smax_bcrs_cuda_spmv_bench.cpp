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
    IT hpad = cli_args->_hpad;
    IT wpad = cli_args->_wpad;
    bool use_cm = cli_args->_use_cm;
    std::string custom_kernel = cli_args->_ck;

    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *y = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel name to SMAX
    std::string bench_name = "smax_bcrs_cuda_spmv";
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPMV,
                            SMAX::PlatformType::CUDA);

    // Declare bcrs operand
    BCRSMatrix<IT, VT> *A_bcrs = new BCRSMatrix<IT, VT>();

    // Convert CRS matrix to BCRS
    smax->utils->convert_crs_to_bcrs<IT, VT, ULL>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, A_bcrs->n_rows, A_bcrs->n_cols,
        A_bcrs->n_blocks, A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
        A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr, A_bcrs->val, hpad, wpad,
        hpad, wpad, use_cm);

    smax->kernel(bench_name)->set_mat_bcrs(true);
    smax->kernel(bench_name)->set_block_column_major(use_cm);

    SMAX::SpMVType custom_kernel_type = SMAX::SpMVType::naive_thread_per_row;

    if (custom_kernel == "tpr") {
        custom_kernel_type = SMAX::SpMVType::naive_thread_per_row;
    } else if (custom_kernel == "nws") {
        custom_kernel_type = SMAX::SpMVType::naive_warp_shuffle;
    } else if (custom_kernel == "nwg") {
        custom_kernel_type = SMAX::SpMVType::naive_warp_group;
    }

    smax->kernel(bench_name)->set_kernel_implementation(custom_kernel_type);

    // A is assumed to be in BCRS format
    smax->kernel(bench_name)
        ->register_A(A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
                     A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
                     A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr,
                     A_bcrs->val);
    // x and y are dense matrices
    smax->kernel(bench_name)->register_B(A_bcrs->n_cols, x->val);
    smax->kernel(bench_name)->register_C(A_bcrs->n_rows, y->val);

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
    delete A_bcrs;
    delete smax;
}