#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    // Set datatypes
    using IT = unsigned int;
    using VT = double;

    // Just takes pinning overhead away from timers
    init_pin();

    // Setup data structures
    INIT_SPMV(IT, VT);
    IT A_scs_C = _C;         // Defined at runtime
    IT A_scs_sigma = _sigma; // Defined at runtime
    IT A_scs_n_rows = 0;
    IT A_scs_n_rows_padded = 0;
    IT A_scs_n_cols = 0;
    IT A_scs_n_chunks = 0;
    IT A_scs_n_elements = 0;
    IT A_scs_nnz = 0;
    IT *A_scs_chunk_ptr = nullptr;
    IT *A_scs_chunk_lengths = nullptr;
    IT *A_scs_col = nullptr;
    VT *A_scs_val = nullptr;
    IT *A_scs_perm = nullptr;

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Convert CRS to Sell-C-sigma matrix
    smax->utils->convert_crs_to_scs<IT, VT, IT>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, A_scs_C, A_scs_sigma, A_scs_n_rows,
        A_scs_n_rows_padded, A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements,
        A_scs_nnz, A_scs_chunk_ptr, A_scs_chunk_lengths, A_scs_col, A_scs_val,
        A_scs_perm);

    // Pad to the same length to emulate real iterative schemes
    IT vec_size = std::max(A_scs_n_rows_padded, A_scs_n_cols);
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
        ->register_A(A_scs_C, A_scs_sigma, A_scs_n_rows, A_scs_n_rows_padded,
                     A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements, A_scs_nnz,
                     A_scs_chunk_ptr, A_scs_chunk_lengths, A_scs_col, A_scs_val,
                     A_scs_perm);
    smax->kernel(bench_name)->register_B(vec_size, x->val);
    smax->kernel(bench_name)->register_C(vec_size, y->val);

    // Setup benchmark harness
    SETUP_BENCH(bench_name);
    std::function<void()> lambda = [smax, bench_name]() {
        smax->kernel(bench_name)->apply();
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMV_BENCH;
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPMV;
    delete[] A_scs_chunk_ptr;
    delete[] A_scs_chunk_lengths;
    delete[] A_scs_col;
    delete[] A_scs_val;
    delete[] A_scs_perm;
    delete x;
    delete y;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}