#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

// Set datatypes
using IT = int;
using VT = double;

int main(int argc, char *argv[]) {

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPTRSV_LVL(IT, VT);
    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *b = new DenseMatrix<VT>(crs_mat->n_cols, 1, 0.0);
    int *perm = new int[crs_mat->n_rows];
    int *inv_perm = new int[crs_mat->n_rows];
    CRSMatrix<int, double> *A_perm = new CRSMatrix<int, double>(crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register kernel name to SMAX
    std::string bench_name = "smax_sptrsv_lvl";
    register_kernel<IT, VT>(smax, bench_name, SMAX::KernelType::SPTRSV,
                            SMAX::PlatformType::CPU);

    // Permute matrix
    smax->utils->generate_perm<int>(crs_mat->n_rows, crs_mat->row_ptr,
                                    crs_mat->col, perm, inv_perm, argv[2]);
    smax->kernel(bench_name)->set_mat_perm(true);
    smax->utils->apply_mat_perm<IT, VT>(
        crs_mat->n_rows, crs_mat->row_ptr, crs_mat->col, crs_mat->val,
        A_perm->row_ptr, A_perm->col, A_perm->val, perm, inv_perm);
    extract_D_L_U<IT, VT>(*A_perm, *crs_mat_D_plus_L, *crs_mat_U);

    // Register kenel data
    REGISTER_SPTRSV_DATA(bench_name, crs_mat_D_plus_L, x, b);

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
    PRINT_SPTRSV_BENCH;
    smax->utils->print_timers();

    // Clean up
    FINALIZE_SPTRSV;
    delete x;
    delete b;
    delete A_perm;
    FINALIZE_LIKWID_MARKERS;
}
