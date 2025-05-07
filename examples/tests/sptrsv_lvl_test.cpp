/**
 * @file
 * @brief Example demonstrating how to use the SMAX library to perform
 * sparse triangular solve with a lower triangular matrix (SpTSV) with
 * permutation and level-set scheduling.
 */
#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

int main(void) {
    // Initialize operands
    int A_n_rows = 3;
    int A_n_cols = 3;
    int A_nnz = 4;
    int *A_col = new int[A_nnz]{0, 1, 0, 2};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 1, 2, 4};
    double *A_val = new double[A_nnz]{1.1, 2.2, 3.1, 3.3};

    double *x = new double[A_n_cols];
    for (int i = 0; i < A_n_cols; ++i) {
        x[i] = 1.0;
    }

    // Initialize RHS
    double *b = new double[A_n_cols];
    for (int i = 0; i < A_n_cols; ++i) {
        b[i] = 2.0;
    }

    // Declare permuted data
    int *A_perm_col = new int[A_nnz];
    int *A_perm_row_ptr = new int[A_n_rows + 1];
    double *A_perm_val = new double[A_nnz];
    double *x_perm = new double[A_n_cols];
    double *b_perm = new double[A_n_cols];

    // Declare permutation vectors
    int *perm = new int[A_n_rows];
    int *inv_perm = new int[A_n_rows];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Get BFS permutation vector
    smax->utils->generate_perm<int>(A_n_rows, A_row_ptr, A_col, perm, inv_perm);

    // Apply permutation vector to A
    smax->utils->apply_mat_perm<int, double>(A_n_rows, A_row_ptr, A_col, A_val,
                                             A_perm_row_ptr, A_perm_col,
                                             A_perm_val, perm);

    // Apply permutation vector to x and b
    smax->utils->apply_vec_perm<double>(A_n_cols, x, x_perm, perm);
    smax->utils->apply_vec_perm<double>(A_n_cols, b, b_perm, perm);

    // Declare L and U data
    int L_n_rows = 0;
    int L_n_cols = 0;
    int L_nnz = 0;
    int *L_col = nullptr;
    int *L_row_ptr = nullptr;
    double *L_val = nullptr;
    int U_n_rows = 0;
    int U_n_cols = 0;
    int U_nnz = 0;
    int *U_col = nullptr;
    int *U_row_ptr = nullptr;
    double *U_val = nullptr;

    extract_D_L_U_arrays(A_n_rows, A_n_cols, A_nnz, A_perm_col, A_perm_row_ptr,
                         A_perm_val, L_n_rows, L_n_cols, L_nnz, L_col,
                         L_row_ptr, L_val, U_n_rows, U_n_cols, U_nnz, U_col,
                         U_row_ptr, U_val);

    // Register kernel tag, platform, and metadata
    smax->register_kernel("solve_perm_Lx=b", SMAX::SPTRSV, SMAX::CPU);

    // Tell SMAX to expect a permuted matrix
    // This enables level-set scheduling for SpTRSV
    smax->kernels["solve_perm_Lx=b"]->set_perm(true);

    // Register operands to this kernel tag
    // A is assumed to be in CRS format
    smax->kernels["solve_perm_Lx=b"]->register_A(L_n_rows, L_n_cols, L_nnz,
                                                 &L_col, &L_row_ptr, &L_val);

    // x and b are dense vectors
    smax->kernels["solve_perm_Lx=b"]->register_B(A_n_rows, &x_perm);
    smax->kernels["solve_perm_Lx=b"]->register_C(A_n_cols, &b_perm);

    // Execute all phases of this kernel
    smax->kernels["solve_perm_Lx=b"]->run();

    // Unpermute solution vector
    smax->utils->apply_vec_perm<double>(A_n_cols, x_perm, x, perm);

    smax->print_timers();

    print_vector<double>(x, A_n_cols);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete[] A_perm_col;
    delete[] A_perm_row_ptr;
    delete[] A_perm_val;
    delete[] perm;
    delete[] inv_perm;
    delete[] x;
    delete[] b;
    delete[] x_perm;
    delete[] b_perm;
    delete smax;

    return 0;
}