#include "../../examples_common.hpp"
#include "../../sptrsv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "petsc_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    //     INIT_SPTRSV;

    //     PetscErrorCode ierr;
    //     ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // Create PETSc vectors for b and x
    //     Vec b, x;
    //     ierr = VecCreate(PETSC_COMM_SELF, &b);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSetSizes(b, PETSC_DECIDE, crs_mat->n_cols);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSetFromOptions(b);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSet(b, 1.0); // Set b to 1.0 for all entries
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     ierr = VecCreate(PETSC_COMM_SELF, &x);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSetSizes(x, PETSC_DECIDE, crs_mat->n_rows);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSetFromOptions(x);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = VecSet(x, 0.0); // Initialize x to 0.0
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // Wrap the CRS matrix in a PETSc matrix
    //     Mat D_plus_L;
    //     ierr = MatCreateSeqAIJWithArrays(
    //         PETSC_COMM_SELF, crs_mat_D_plus_L->n_rows,
    //         crs_mat_D_plus_L->n_cols, crs_mat_D_plus_L->row_ptr,
    //         crs_mat_D_plus_L->col, crs_mat_D_plus_L->val, &D_plus_L);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // // Create a factored version of A for triangular solve
    //     // Mat A_fact;
    //     // ierr = MatDuplicate(A, MAT_COPY_VALUES, &A_fact);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // // Perform LU factorization (symbolic + numeric)
    //     // MatFactorInfo factor_info;
    //     // ierr = MatFactorInfoInitialize(&factor_info);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // IS row_perm, col_perm;
    //     // ierr = MatGetOrdering(A_fact, MATORDERINGNATURAL, &row_perm,
    //     &col_perm);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // ierr = MatLUFactor(A_fact, row_perm, col_perm, &factor_info);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // ierr = ISDestroy(&row_perm);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     // ierr = ISDestroy(&col_perm);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     ierr = MatSetOption(D_plus_L, MAT_STRUCTURALLY_SYMMETRIC,
    //     PETSC_TRUE); CHKERRABORT(PETSC_COMM_SELF, ierr); ierr =
    //     MatSetOption(D_plus_L, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = MatSetOption(D_plus_L, MAT_FACTOR_LU, PETSC_FALSE);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     ierr = MatSetOption(D_plus_L, MAT_FACTOR_SHIFT_TYPE, MAT_SHIFT_NONE);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     MatFactorInfo info;
    //     ierr = MatFactorInfoInitialize(&info);
    //     CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // IS row_perm, col_perm;
    //     // ierr = MatGetOrdering(A, MATORDERINGNATURAL, &row_perm,
    //     &col_perm);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // // This performs only symbolic 'L' factor, assuming A is D+L
    //     // ierr =
    //     //     MatILUFactor(A, row_perm, col_perm, &info); // or MatLUFactor
    //     if
    //     //     needed
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);

    //     // ierr = ISDestroy(&row_perm);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);
    //     // ierr = ISDestroy(&col_perm);
    //     // CHKERRABORT(PETSC_COMM_SELF, ierr);

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
    //     std::function<void(bool)> lambda = [bench_name, D_plus_L, b,
    //                                         x](bool warmup) {
    //         IF_USE_LIKWID(if (!warmup)
    //         LIKWID_MARKER_START(bench_name.c_str());)

    //         // Solve A * x = b, using factored matrix
    //         MatSolve(D_plus_L, b, x);

    //         IF_USE_LIKWID(if (!warmup)
    //         LIKWID_MARKER_STOP(bench_name.c_str());)
    //     };

    //     RUN_BENCH;
    //     PRINT_SPTRSV_BENCH;
    //     FINALIZE_SPTRSV;
    //     delete bench_harness;

    //     // Cleanup PETSc objects
    //     ierr = MatDestroy(&D_plus_L);
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