#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "petsc_benchmarks_common.hpp"

int main(int argc, char *argv[]) {
    INIT_SPMM;

    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    int m = crs_mat->n_rows;
    int n = crs_mat->n_cols;
    int k = n_vectors;

    // Create dense matrices X (input) and Y (output)
    Mat X, Y;
    ierr = MatCreateDense(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, k,
                          NULL, &X);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = MatSetOption(X, MAT_ROW_ORIENTED, PETSC_TRUE);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = MatZeroEntries(X);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    PetscScalar *X_array;
    ierr = MatDenseGetArray(X, &X_array);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    for (int i = 0; i < n * k; i++)
        X_array[i] = 1.0;
    ierr = MatDenseRestoreArray(X, &X_array);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    // Wrap CRS matrix in PETSc
    Mat A;
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, n, crs_mat->row_ptr,
                                     crs_mat->col, crs_mat->values, &A);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    std::string bench_name = "petsc_spmm";
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

    init_pin();

    // Perform initial matrix-matrix multiplication to allocate Y
    ierr = MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    std::function<void(bool)> lambda = [bench_name, A, X, &Y](bool warmup) {
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_START(bench_name.c_str());)
        MatMatMult(A, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y);
        IF_USE_LIKWID(if (!warmup) LIKWID_MARKER_STOP(bench_name.c_str());)
    };

    RUN_BENCH;
    PRINT_SPMM_BENCH;
    FINALIZE_SPMM;
    delete bench_harness;

    ierr = MatDestroy(&A);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = MatDestroy(&X);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = MatDestroy(&Y);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = PetscFinalize();
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    return 0;
}