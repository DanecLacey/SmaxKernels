#include "utils.hpp"
#include "SmaxKernels/interface.hpp"

#include <iostream>

#define TOL 1e-6
#define MAX_ITERS 1000
#define CHECK_LENGTH 5

int main(void){
    int n = 6; // Total points including boundaries
    CRSMatrix A = create1DPoissonMatrixCRS(n);
    DenseMatrix x = createDenseMatrix(A.n_cols, 0.0);
    DenseMatrix b = createDenseMatrix(A.n_cols, 1.0);
    DenseMatrix tmp = createDenseMatrix(A.n_cols, 0.0);
    DenseMatrix residual = createDenseMatrix(A.n_cols, 0.0);

	CRSMatrix D_plus_L{};
	CRSMatrix U{};
    extract_D_L_U(A, D_plus_L, U);

    // printCRSMatrix(D_plus_L);
    // printCRSMatrix(U);
    // exit(0);

    // tmp <- Ax
    // SMAX::Interface *Ax_spmv = new SMAX::Interface(SMAX::SPMV, SMAX::CPU);
    // Ax_spmv->register_A(&A.n_rows, &A.n_cols, &A.nnz, A.col.data(), A.row_ptr.data(), A.values.data()); // A
    // Ax_spmv->register_B(&A.n_cols, &x.n_cols, x.values.data()); // X
    // Ax_spmv->register_C(&A.n_cols, &tmp.n_cols, tmp.values.data()); // tmp

    // // tmp <- Ux
    // SMAX::Interface *Ux_spmv = new SMAX::Interface(SMAX::SPMV, SMAX::CPU);
    // Ux_spmv->register_A(&U.n_rows, &U.n_cols, &U.nnz, U.col.data(), U.row_ptr.data(), U.values.data()); // U
    // Ux_spmv->register_B(&U.n_cols, &x.n_cols, x.values.data()); // X
    // Ux_spmv->register_C(&U.n_cols, &tmp.n_cols, tmp.values.data()); // tmp

    // (D+L)x <- b - Ux
    // TODO
    // SMAX::Interface *sptsv = new SMAX::Interface(SMAX::SPTSV, SMAX::CPU);
    // sptsv->register_A(&D_plus_L.n_rows, &D_plus_L.n_cols, &D_plus_L.nnz, D_plus_L.col.data(), D_plus_L.row_ptr.data(), D_plus_L.values.data());
    // sptsv->register_B(&A.n_cols, &x.n_cols, x.values.data());
    // sptsv->register_C(&tmp.n_cols, &tmp.n_cols, tmp.values.data());

    
    double residual_norm = 0.0;
    // Compute initial residual norm
    // Ax_spmv->run();
    spmv(A, x.values, tmp.values);
    subtract_vectors(residual.values.data(), b.values.data(), tmp.values.data(), b.n_rows);
    residual_norm = infty_vec_norm(residual);
    std::cout << "Initial residual norm: " << residual_norm << std::endl;

    int n_iters = 0;
    do {
        // tmp <- Ux
        // Ux_spmv->run();
        spmv(U, x.values, tmp.values);

        // tmp <- b - Ux
        subtract_vectors(tmp.values.data(), b.values.data(), tmp.values.data(), tmp.n_rows);

        // x <- (D+L)^{-1}(b - Ux)
        // sptsv->run(); TODO
        spltsv(D_plus_L, b.values, x.values);

        if(n_iters % CHECK_LENGTH == 0){
            // Compute residual

            // tmp <- Ax
            // Ax_spmv->run();
            spmv(A, x.values, tmp.values);
            // residual <- b - Ax
            subtract_vectors(residual.values.data(), b.values.data(), tmp.values.data(), tmp.n_rows);
            // residual_norm = || b - Ax ||
            residual_norm = infty_vec_norm(residual);
        }
        ++n_iters;

    } while (residual_norm > TOL || n_iters < MAX_ITERS);

    if(infty_vec_norm(residual) < TOL){
        std::cout << "Gauss-Seidel solver converged after " << n_iters << " iterations." << std::endl;
    }
    else{
        std::cout << "Gauss-Seidel solver did not converge." << std::endl;
    }

    // delete Ax_spmv;
    // delete Ux_spmv;
     
    return 0;
}