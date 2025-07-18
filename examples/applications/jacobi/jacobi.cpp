#include "../../examples_common.hpp"
#include "../applications_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

#include <iostream>

#define TOL 1e-8
#define MAX_ITERS 1000
#define CHECK_LENGTH 5
#define DOMAIN_SIZE 10
#define X_INIT 1.0

double check_residual(DenseMatrix<double> *b, DenseMatrix<double> *tmp,
                      DenseMatrix<double> *residual, SMAX::Interface *smax) {

    smax->kernel("tmp <- Ax")->run();

    // residual <- (b - Ax)
    subtract_vectors(residual->val, b->val, tmp->val, b->n_rows);

    // residual_norm = || b - Ax ||
    return infty_vec_norm(residual->val, b->n_rows);
}

void jacobi_iter(DenseMatrix<double> *x_old, DenseMatrix<double> *x_new,
                 DenseMatrix<double> *b, DenseMatrix<double> *D,
                 SMAX::Interface *smax) {

    smax->kernel("x_new <- Ax_old")->run();

    normalize_x(x_new, x_old, D, b);
}

void solve(DenseMatrix<double> *x_old, DenseMatrix<double> *x_new,
           DenseMatrix<double> *b, DenseMatrix<double> *tmp,
           DenseMatrix<double> *residual, DenseMatrix<double> *D, int &n_iters,
           double &residual_norm, SMAX::Interface *smax) {

    while (residual_norm > TOL && n_iters < MAX_ITERS) {

        jacobi_iter(x_old, x_new, b, D, smax);

        // Swap library pointers and application pointers to keep in-sync
        // This is because SMAX does not own the memory at x_old and x_new,
        // it merely points to the same memory locations
        smax->kernel("x_new <- Ax_old")->swap_operands();
        std::swap(x_old, x_new);

        ++n_iters;

        // Compute residual every CHECK_LENGTH iterations
        if (n_iters % CHECK_LENGTH == 0) {
            residual_norm = check_residual(b, tmp, residual, smax);
            PRINT_ITER(n_iters, residual_norm);
        }
    }
}

int main(void) {
#if SMAX_CUDA_MODE
    printf("Using CUDA kernels.\n");
    constexpr SMAX::PlatformType Platform = SMAX::PlatformType::CUDA;
#else
    printf("Using CPU kernels.\n");
    constexpr SMAX::PlatformType Platform = SMAX::PlatformType::CPU;
#endif

    // Set up problem
    CRSMatrix<int, double> *A = create2DPoissonMatrixCRS(DOMAIN_SIZE);
    CRSMatrix<int, double> *A_perm =
        new CRSMatrix<int, double>(A->n_rows, A->n_cols, A->nnz);
    DenseMatrix<double> *x_new = new DenseMatrix<double>(A->n_cols, 1, X_INIT);
    DenseMatrix<double> *x_old = new DenseMatrix<double>(A->n_cols, 1, X_INIT);
    DenseMatrix<double> *b = new DenseMatrix<double>(A->n_cols, 1, 1.0);
    DenseMatrix<double> *D = new DenseMatrix<double>(A->n_cols, 1, 1.0);
    DenseMatrix<double> *tmp = new DenseMatrix<double>(A->n_cols, 1, 0.0);
    DenseMatrix<double> *residual = new DenseMatrix<double>(A->n_cols, 1, 0.0);

    int *perm = new int[A->n_rows];
    int *inv_perm = new int[A->n_rows];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    smax->utils->generate_perm(A->n_rows, A->row_ptr, A->col, perm, inv_perm);
    PRINT_PERM_VECTOR(A, perm);

    smax->utils->apply_mat_perm<int, double>(
        A->n_rows, A->row_ptr, A->col, A->val, A_perm->row_ptr, A_perm->col,
        A_perm->val, perm, inv_perm);

    peel_diag_crs(A_perm, D);

    // Register necessary sparse kernels to SMAX
    smax->register_kernel("tmp <- Ax", SMAX::KernelType::SPMV, Platform);
    REGISTER_SPMV_DATA("tmp <- Ax", A_perm, x_new, tmp);

    smax->register_kernel("x_new <- Ax_old", SMAX::KernelType::SPMV, Platform);
    REGISTER_SPMV_DATA("x_new <- Ax_old", A_perm, x_old, x_new);

    // Compute initial residual norm
    double residual_norm = check_residual(b, tmp, residual, smax);

    // Iterate until convergence is reached
    int n_iters = 0;
    PRINT_ITER(n_iters, residual_norm);
    solve(x_old, x_new, b, tmp, residual, D, n_iters, residual_norm, smax);

    if (residual_norm < TOL) {
        std::cout << "Jacobi solver converged after " << n_iters
                  << " iterations." << std::endl;
    } else {
        std::cout << "Jacobi solver did not converge." << std::endl;
    }
    std::cout << "Final residual norm: " << residual_norm << std::endl;

    smax->utils->print_timers();

    delete A_perm;
    delete x_new;
    delete x_old;
    delete b;
    delete tmp;
    delete residual;
    delete D;
    delete smax;

    return 0;
}