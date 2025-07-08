#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    using IT = unsigned int;
    using VT = double;

    // Just to take overhead of pinning away from timers
    init_pin();

    INIT_SPMV(IT, VT);
    // Declare Sell-c-sigma operand
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

    std::ostringstream oss;
    oss << "my_SELL_" << _C << "_" << _sigma << "_spmv";
    std::string kernel_name = oss.str();

    register_kernel<IT, VT>(smax, std::string(kernel_name),
                            SMAX::KernelType::SPMV, SMAX::PlatformType::CPU);

    // A is expected to be in the SCS format
    smax->kernel(kernel_name)->set_mat_scs(true);

    smax->kernel(kernel_name)
        ->register_A(A_scs_C, A_scs_sigma, A_scs_n_rows, A_scs_n_rows_padded,
                     A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements, A_scs_nnz,
                     A_scs_chunk_ptr, A_scs_chunk_lengths, A_scs_col, A_scs_val,
                     A_scs_perm);
    smax->kernel(kernel_name)->register_B(vec_size, x->val);
    smax->kernel(kernel_name)->register_C(vec_size, y->val);

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = kernel_name;
    float runtime = 0.0;
    int n_iter = MIN_NUM_ITERS;
    int n_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
#endif

#ifdef USE_LIKWID
    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
        LIKWID_MARKER_REGISTER(bench_name.c_str());
    }
#endif

    std::function<void(bool)> lambda = [bench_name, smax,
                                        kernel_name](bool warmup) {
        PARALLEL_LIKWID_MARKER_START(bench_name.c_str());
        smax->kernel(kernel_name)->run();
        PARALLEL_LIKWID_MARKER_STOP(bench_name.c_str());
    };

    RUN_BENCH;
    PRINT_SPMV_BENCH;

    smax->utils->print_timers();

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