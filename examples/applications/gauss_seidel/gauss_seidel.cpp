#include "../../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

#include <iostream>

#define TOL 1e-8
#define MAX_ITERS 1000
#define CHECK_LENGTH 5
#define DOMAIN_SIZE 10

#ifndef DEBUG
#define DEBUG 1
#endif

double check_residual(double *b, double *tmp, double *residual, int n_rows,
                      SMAX::Interface *smax) {
    smax->kernels["tmp <- Ax"]->run();

    // residual <- (b - Ax)
    subtract_vectors(residual, b, tmp, n_rows);

    // residual_norm = || b - Ax ||
    return infty_vec_norm(residual, n_rows);
}

void gauss_seidel_iter(double *b, double *tmp, int n_rows,
                       SMAX::Interface *smax) {
    smax->kernels["tmp <- Ux"]->run();

    // tmp <- (b - Ux)
    subtract_vectors(tmp, b, tmp, n_rows);

    smax->kernels["solve x <- (D+L)^{-1}(b-Ux)"]->run();
}

void solve(double *b, double *tmp, double *residual, int n_rows, int &n_iters,
           double &residual_norm, SMAX::Interface *smax) {
    do {
        gauss_seidel_iter(b, tmp, n_rows, smax);

        // Compute residual every CHECK_LENGTH iterations
        if (n_iters % CHECK_LENGTH == 0) {
            residual_norm = check_residual(b, tmp, residual, n_rows, smax);
            if (DEBUG) {
                printf("iter: %d, residual_norm = %f\n", n_iters,
                       residual_norm);
            }
        }
        ++n_iters;

    } while (residual_norm > TOL && n_iters < MAX_ITERS);
}

int main(void) {
    // Set up problem
    CRSMatrix *A = create1DPoissonMatrixCRS(DOMAIN_SIZE);
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
    REGISTER_SPMV_KERNEL("tmp <- Ax", A, x, tmp);
    REGISTER_SPMV_KERNEL("tmp <- Ux", U, x, tmp);
    REGISTER_SPTRSV_KERNEL("solve x <- (D+L)^{-1}(b-Ux)", D_plus_L, x, tmp);

    // Compute initial residual norm
    double residual_norm = check_residual(b->values, tmp->values,
                                          residual->values, b->n_rows, smax);

    // Iterate until convergence is reached
    int n_iters = 0;
    solve(b->values, tmp->values, residual->values, b->n_rows, n_iters,
          residual_norm, smax);

    if (residual_norm < TOL) {
        std::cout << "Gauss-Seidel solver converged after " << n_iters
                  << " iterations." << std::endl;
    } else {
        std::cout << "Gauss-Seidel solver did not converge." << std::endl;
    }
    std::cout << "Final residual norm: " << residual_norm << std::endl;

    delete x;
    delete b;
    delete tmp;
    delete residual;
    delete smax;
    delete U;
    delete D_plus_L;
    return 0;
}