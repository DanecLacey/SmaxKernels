#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "smax_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPTRSV;
    DenseMatrix *x = new DenseMatrix(crs_mat->n_cols, 1, 1.0);
    DenseMatrix *b = new DenseMatrix(crs_mat->n_cols, 1, 0.0);
    int *perm = new int[crs_mat->n_rows];
    int *inv_perm = new int[crs_mat->n_rows];
    int *A_perm_col = new int[crs_mat->nnz];
    int *A_perm_row_ptr = new int[crs_mat->n_rows + 1];
    double *A_perm_val = new double[crs_mat->nnz];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    smax->register_kernel("my_sptrsv_lvl", SMAX::KernelType::SPTRSV);
    smax->utils->generate_perm_jh<int>(crs_mat->n_rows, crs_mat->row_ptr,
                                       crs_mat->col, perm, inv_perm);
    smax->kernel("my_sptrsv_lvl")->set_mat_perm(true);
    smax->utils->apply_mat_perm<int, double>(
        crs_mat->n_rows, crs_mat->row_ptr, crs_mat->col, crs_mat->val,
        A_perm_row_ptr, A_perm_col, A_perm_val, perm, inv_perm);
    extract_D_L_U_arrays(crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz,
                         A_perm_row_ptr, A_perm_col, A_perm_val,
                         crs_mat_D_plus_L->n_rows, crs_mat_D_plus_L->n_cols,
                         crs_mat_D_plus_L->nnz, crs_mat_D_plus_L->row_ptr,
                         crs_mat_D_plus_L->col, crs_mat_D_plus_L->val,
                         crs_mat_U->n_rows, crs_mat_U->n_cols, crs_mat_U->nnz,
                         crs_mat_U->row_ptr, crs_mat_U->col, crs_mat_U->val);
    REGISTER_SPTRSV_DATA("my_sptrsv_lvl", crs_mat_D_plus_L, x, b);

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "smax_sptrsv_lvl";
    double runtime = 0.0;
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

    // Just to take overhead of pinning away from timers
    init_pin();

    std::function<void(bool)> lambda = [bench_name, smax](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        smax->kernel("my_sptrsv_lvl")->run();
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPTRSV_BENCH;
    FINALIZE_SPTRSV;
    delete bench_harness;
    delete x;
    delete b;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}
