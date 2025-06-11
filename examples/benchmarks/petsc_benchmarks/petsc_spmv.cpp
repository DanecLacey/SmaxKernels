#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "petsc_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    init_pin(); // avoid counting pinning in timing

    INIT_SPMV;

    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    Vec x, y;
    ierr = VecCreateSeq(PETSC_COMM_SELF, crs_mat->n_cols, &x);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = VecSet(x, 1.0);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF, crs_mat->n_rows, &y);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = VecSet(y, 0.0);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    // Wrap the CRS matrix in PETSc
    Mat A;
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, crs_mat->n_rows,
                                     crs_mat->n_cols, crs_mat->row_ptr,
                                     crs_mat->col, crs_mat->val, &A);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    // Make lambda, and pass to the benchmarking harness
    std::string bench_name = "petsc_spmv";
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

    std::function<void(bool)> lambda = [bench_name, A, x, y](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        MatMult(A, x, y);
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPMV_BENCH;
    FINALIZE_SPMV;
    delete bench_harness;

    ierr = VecDestroy(&x);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = VecDestroy(&y);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = MatDestroy(&A);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = PetscFinalize();
    CHKERRABORT(PETSC_COMM_SELF, ierr);

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}