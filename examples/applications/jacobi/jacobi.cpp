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

#ifndef DEBUG
#define DEBUG 1
#endif

double check_residual(DenseMatrix *b, DenseMatrix *tmp, DenseMatrix *residual,
                      SMAX::Interface *smax) {

    smax->kernel("tmp <- Ax")->run();

    // residual <- (b - Ax)
    subtract_vectors(residual->val, b->val, tmp->val, b->n_rows);

    // residual_norm = || b - Ax ||
    return infty_vec_norm(residual->val, b->n_rows);
}

void jacobi_iter(DenseMatrix *x_old, DenseMatrix *x_new, DenseMatrix *b,
                 DenseMatrix *D, SMAX::Interface *smax) {

    smax->kernel("x_new <- Ax_old")->run();

    normalize_x(x_new, x_old, D, b);
}

void solve(DenseMatrix *x_old, DenseMatrix *x_new, DenseMatrix *b,
           DenseMatrix *tmp, DenseMatrix *residual, DenseMatrix *D,
           int &n_iters, double &residual_norm, SMAX::Interface *smax) {

    while (residual_norm > TOL && n_iters < MAX_ITERS) {

        jacobi_iter(x_old, x_new, b, D, smax);

        smax->kernel("x_new <- Ax_old")->swap_operands();

        ++n_iters;

        // Compute residual every CHECK_LENGTH iterations
        if (n_iters % CHECK_LENGTH == 0) {
            residual_norm = check_residual(b, tmp, residual, smax);
            DEBUG_PRINT_ITER(n_iters, residual_norm);
        }
    }
}

int main(void) {
    // Set up problem
    CRSMatrix *A = create2DPoissonMatrixCRS(DOMAIN_SIZE);
    CRSMatrix *A_perm = new CRSMatrix(A->n_rows, A->n_cols, A->nnz);
    DenseMatrix *x_new = new DenseMatrix(A->n_cols, 1, X_INIT);
    DenseMatrix *x_old = new DenseMatrix(A->n_cols, 1, X_INIT);
    DenseMatrix *b = new DenseMatrix(A->n_cols, 1, 1.0);
    DenseMatrix *D = new DenseMatrix(A->n_cols, 1, 1.0);
    DenseMatrix *tmp = new DenseMatrix(A->n_cols, 1, 0.0);
    DenseMatrix *residual = new DenseMatrix(A->n_cols, 1, 0.0);

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
    smax->register_kernel("tmp <- Ax", SMAX::SPMV, SMAX::CPU);
    REGISTER_SPMV_DATA("tmp <- Ax", A_perm, x_new, tmp);

    smax->register_kernel("x_new <- Ax_old", SMAX::SPMV, SMAX::CPU);
    REGISTER_SPMV_DATA("x_new <- Ax_old", A_perm, x_old, x_new);

    // Compute initial residual norm
    double residual_norm = check_residual(b, tmp, residual, smax);

    // Iterate until convergence is reached
    int n_iters = 0;
    DEBUG_PRINT_ITER(n_iters, residual_norm);
    solve(x_old, x_new, b, tmp, residual, D, n_iters, residual_norm, smax);

    if (residual_norm < TOL) {
        std::cout << "Jacobi solver converged after " << n_iters
                  << " iterations." << std::endl;
    } else {
        std::cout << "Jacobi solver did not converge." << std::endl;
    }
    std::cout << "Final residual norm: " << residual_norm << std::endl;

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