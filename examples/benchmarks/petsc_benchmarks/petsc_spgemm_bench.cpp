#include "../../examples_common.hpp"
#include "../../spgemm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "petsc_benchmarks_common.hpp"

int main(int argc, char *argv[]) {

    init_pin(); // avoid counting pinning in timing

    INIT_SPGEMM;

    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    // --- 1) Wrap CRS A and B in PETSc SeqAIJ matrices ---
    Mat A, B, C;
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, crs_mat_A->n_rows,
                                     crs_mat_A->n_cols, crs_mat_A->row_ptr,
                                     crs_mat_A->col, crs_mat_A->val, &A);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, crs_mat_B->n_rows,
                                     crs_mat_B->n_cols, crs_mat_B->row_ptr,
                                     crs_mat_B->col, crs_mat_B->val, &B);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    // --- 2) Create C once with an initial multiplication ---
    ierr = MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    // --- 3) Benchmark metadata ---
    std::string bench_name = "petsc_spgemm";
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

    // --- 4) The kernel: reuse C and re‐compute A*B ---
    std::function<void(bool)> lambda = [bench_name, A, B, &C](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)

        // MAT_REUSE_MATRIX tells PETSc to reuse C’s sparsity structure
        MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C);

        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    MatInfo info;
    ierr = MatGetInfo(C, MAT_LOCAL, &info);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    PetscInt C_nnz = (PetscInt)info.nz_used;

    RUN_BENCH;
    PRINT_SPGEMM_BENCH(C_nnz);

    // --- 5) Cleanup ---
    ierr = MatDestroy(&A);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = MatDestroy(&B);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = MatDestroy(&C);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    FINALIZE_SPGEMM;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

    ierr = PetscFinalize();
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    return 0;
}