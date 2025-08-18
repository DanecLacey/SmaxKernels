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
    std::string out = cli_args->_out;

    ULL n_rows = crs_mat->n_rows;
    int *perm = new int[n_rows];
    int *inv_perm = new int[n_rows];
    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *y = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel name to SMAX
    std::string bench_name = "smax_crs_spmv";
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPMV,
                            SMAX::PlatformType::CPU);

    // Generate and apply permutation
    bool permute = true;
    if (permute) {
        CRSMatrix<IT, VT> *crs_mat_perm = new CRSMatrix<IT, VT>(
            crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz);

        smax->utils->generate_perm<IT>(crs_mat->n_rows, crs_mat->row_ptr,
                                       crs_mat->col, perm, inv_perm,
                                       std::string("BFS_BW"));
        smax->utils->apply_mat_perm<IT, VT>(
            n_rows, crs_mat->row_ptr, crs_mat->col, crs_mat->val,
            crs_mat_perm->row_ptr, crs_mat_perm->col, crs_mat_perm->val, perm,
            inv_perm);

        crs_mat = crs_mat_perm;
    }

    // Register kenel data
    REGISTER_SPMV_DATA(bench_name, crs_mat, x, y);

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
    delete x;
    delete y;
    delete perm;
    delete inv_perm;
    FINALIZE_LIKWID_MARKERS;
}