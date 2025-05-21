#include "../../examples_common.hpp"
#include "../applications_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

#include <iostream>

#define TOL 1e-8
#define MAX_ITERS 1000
#define CHECK_LENGTH 5
#define DOMAIN_SIZE 10

double check_residual(DenseMatrix *b, DenseMatrix *tmp, DenseMatrix *residual,
                      SMAX::Interface *smax) {

    smax->kernel("tmp <- Ax")->run();

    // residual <- (b - Ax)
    subtract_vectors(residual->val, b->val, tmp->val, b->n_rows);

    // residual_norm = || b - Ax ||
    return infty_vec_norm(residual->val, b->n_rows);
}

void gauss_seidel_iter(DenseMatrix *b, DenseMatrix *tmp,
                       SMAX::Interface *smax) {

    smax->kernel("tmp <- Ux")->run();

    // tmp <- (b - Ux)
    subtract_vectors(tmp->val, b->val, tmp->val, b->n_rows);

    smax->kernel("solve x <- (D+L)^{-1}(b-Ux)")->run();
}

void solve(DenseMatrix *b, DenseMatrix *tmp, DenseMatrix *residual,
           int &n_iters, double &residual_norm, SMAX::Interface *smax) {

    while (residual_norm > TOL && n_iters < MAX_ITERS) {
        gauss_seidel_iter(b, tmp, smax);
        ++n_iters;

        // Compute residual every CHECK_LENGTH iterations
        if (n_iters % CHECK_LENGTH == 0) {
            residual_norm = check_residual(b, tmp, residual, smax);
            PRINT_ITER(n_iters, residual_norm);
        }
    }
}

int main(void) {
#if SMAX_USE_CUDA
    printf("Using CUDA kernels.\n");
    constexpr SMAX::PlatformType Platform = SMAX::PlatformType::CUDA;
#else
    printf("Using CPU kernels.\n");
    constexpr SMAX::PlatformType Platform = SMAX::PlatformType::CPU;
#endif

    // Set up problem
    CRSMatrix *A = create2DPoissonMatrixCRS(DOMAIN_SIZE);
    DenseMatrix *x = new DenseMatrix(A->n_cols, 1, 0.0);
    DenseMatrix *b = new DenseMatrix(A->n_cols, 1, 1.0);
    DenseMatrix *tmp = new DenseMatrix(A->n_cols, 1, 0.0);
    DenseMatrix *residual = new DenseMatrix(A->n_cols, 1, 0.0);
    CRSMatrix *D_plus_L = new CRSMatrix();
    CRSMatrix *U = new CRSMatrix();
    extract_D_L_U(*A, *D_plus_L, *U);

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Register necessary sparse kernels to SMAX
    smax->register_kernel("tmp <- Ax", SMAX::KernelType::SPMV, Platform);
    REGISTER_SPMV_DATA("tmp <- Ax", A, x, tmp);

    smax->register_kernel("tmp <- Ux", SMAX::KernelType::SPMV, Platform);
    REGISTER_SPMV_DATA("tmp <- Ux", U, x, tmp);

    smax->register_kernel("solve x <- (D+L)^{-1}(b-Ux)",
                          SMAX::KernelType::SPTRSV);
    REGISTER_SPTRSV_DATA("solve x <- (D+L)^{-1}(b-Ux)", D_plus_L, x, tmp);

    // Compute initial residual norm
    double residual_norm = check_residual(b, tmp, residual, smax);

    // Iterate until convergence is reached
    int n_iters = 0;
    PRINT_ITER(n_iters, residual_norm);
    solve(b, tmp, residual, n_iters, residual_norm, smax);

    if (residual_norm < TOL) {
        std::cout << "Gauss-Seidel solver converged after " << n_iters
                  << " iterations." << std::endl;
    } else {
        std::cout << "Gauss-Seidel solver did not converge." << std::endl;
    }
    std::cout << "Final residual norm: " << residual_norm << std::endl;

    smax->utils->print_timers();

    delete x;
    delete b;
    delete tmp;
    delete residual;
    delete smax;
    delete U;
    delete D_plus_L;
    return 0;
}