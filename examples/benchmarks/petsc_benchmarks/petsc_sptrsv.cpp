#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "petsc_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    // DL 05.05.25 TODO: Not working

    //     INIT_SPTRSV;

    //     PetscErrorCode ierr;
    //     ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // Create PETSc vectors for b and x
    //     Vec b, x;
    //     ierr = VecCreate(PETSC_COMM_WORLD, &b);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSetSizes(b, PETSC_DECIDE, crs_mat->n_cols);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSetFromOptions(b);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSet(b, 1.0); // Set b to 1.0 for all entries
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     ierr = VecCreate(PETSC_COMM_WORLD, &x);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSetSizes(x, PETSC_DECIDE, crs_mat->n_rows);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSetFromOptions(x);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSet(x, 0.0); // Initialize x to 0.0
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // Wrap the CRS matrix in PETSc
    //     Mat A;
    //     ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, crs_mat->n_rows,
    //                                      crs_mat->n_cols, crs_mat->row_ptr,
    //                                      crs_mat->col, crs_mat->val, &A);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // Indicate that the matrix is lower triangular
    //     ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE); // Optional for
    //     symmetric CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // Benchmark Setup
    //     std::string bench_name = "petsc_sptrsv";
    //     double runtime = 0.0;
    //     int n_iter = MIN_NUM_ITERS;
    //     int n_threads = 1;

    // #ifdef _OPENMP
    // #pragma omp parallel
    //     {
    //         n_threads = omp_get_num_threads();
    //     }
    // #endif

    // #ifdef USE_LIKWID
    //     LIKWID_MARKER_INIT;
    // #pragma omp parallel
    //     {
    //         LIKWID_MARKER_REGISTER(bench_name.c_str());
    //     }
    // #endif

    //     init_pin();

    //     // Lambda for benchmarking lower triangular solve
    //     std::function<void(bool)> lambda = [bench_name, A, b, x](bool warmup)
    //     {
    //         IF_USE_LIKWID(if (!warmup)
    //         LIKWID_MARKER_START(bench_name.c_str());)

    //         // Perform forward substitution on the lower triangular part of A
    //         MatSolve(A, b, x); // Use MatSolve to handle the triangular solve

    //         IF_USE_LIKWID(if (!warmup)
    //         LIKWID_MARKER_STOP(bench_name.c_str());)
    //     };

    //     RUN_BENCH;
    //     PRINT_SPTRSV_BENCH;
    //     FINALIZE_SPTRSV;
    //     delete bench_harness;

    //     // Cleanup PETSc objects
    //     ierr = MatDestroy(&A);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecDestroy(&b);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecDestroy(&x);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     ierr = PetscFinalize();
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    // #ifdef USE_LIKWID
    //     LIKWID_MARKER_CLOSE;
    // #endif
    //     return 0;
}