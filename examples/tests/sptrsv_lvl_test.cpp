/**
 * @file
 * @brief Example demonstrating how to use the SMAX library to perform
 * sparse triangular solve with a lower triangular matrix (SpTSV) with
 * BFS permutation and level-set scheduling.
 */

//          Before BFS Permutation
//        0   1   2   3   4   5   6   7
//       _______________________________
//  0   |11                            |
//  1   |    22                        |
//  2   |31      33                    |
//  3   |    42  43  44                |
//  4   |        53      55            |
//  5   |            64      66        |
//  6   |                75      77    |
//  7   |                    86      88|
//       _______________________________

//      After BFS Permutation using A+A^T
//        0   2   4   3   6   1   5   7
//       _______________________________
//  0   |11                            | L0
//  2   |31  33                        | L1
//  4   |    53  55                    | L2
//  3   |    43      44      42        | L3
//  6   |        75      77            | L3
//  1   |                    22        | L4
//  5   |            64          66    | L4
//  7   |                        86  88| L5
//       _______________________________

#include "../examples_common.hpp"
#include "SmaxKernels/interface.hpp"
#include "utils.hpp"

int main(void) {
    // Initialize operands
    int A_n_rows = 8;
    int A_n_cols = 8;
    int A_nnz = 15;
    int *A_col = new int[A_nnz]{0, 1, 0, 2, 1, 2, 3, 2, 4, 3, 5, 4, 6, 5, 7};
    int *A_row_ptr = new int[A_n_rows + 1]{0, 1, 2, 4, 7, 9, 11, 13, 15};
    double *A_val =
        new double[A_nnz]{11.0, 22.0, 31.0, 33.0, 42.0, 43.0, 44.0, 53.0,
                          55.0, 64.0, 66.0, 75.0, 77.0, 86.0, 88.0};

    double *x = new double[A_n_cols];
    for (int i = 0; i < A_n_cols; ++i) {
        x[i] = 1.0;
    }

    // Initialize RHS
    double *b = new double[A_n_rows];
    for (int i = 0; i < A_n_rows; ++i) {
        b[i] = 2.0;
    }

    // Declare permuted data
    int *A_perm_col = new int[A_nnz];
    int *A_perm_row_ptr = new int[A_n_rows + 1];
    double *A_perm_val = new double[A_nnz];
    double *x_perm = new double[A_n_cols];
    double *b_perm = new double[A_n_rows];

    // Declare permutation vectors
    int *perm = new int[A_n_rows];
    int *inv_perm = new int[A_n_rows];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Get BFS permutation vector
    smax->utils->generate_perm<int>(A_n_rows, A_row_ptr, A_col, perm, inv_perm, std::string("BFS"));

    printf("BFS Permutation:\n");
    print_vector<int>(perm, A_n_rows);

    printf("A:\n");
    print_matrix(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);

    // Apply permutation vector to A
    smax->utils->apply_mat_perm<int, double>(A_n_rows, A_row_ptr, A_col, A_val,
                                             A_perm_row_ptr, A_perm_col,
                                             A_perm_val, perm, inv_perm);

    printf("A_perm:\n");
    print_matrix(A_n_rows, A_n_cols, A_nnz, A_perm_col, A_perm_row_ptr,
                 A_perm_val);

    // Apply permutation vector to x and b
    smax->utils->apply_vec_perm<double>(A_n_cols, x, x_perm, perm);
    smax->utils->apply_vec_perm<double>(A_n_rows, b, b_perm, perm);

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

    extract_D_L_U_arrays(A_n_rows, A_n_cols, A_nnz, A_perm_row_ptr, A_perm_col,
                         A_perm_val, L_n_rows, L_n_cols, L_nnz, L_row_ptr,
                         L_col, L_val, U_n_rows, U_n_cols, U_nnz, U_row_ptr,
                         U_col, U_val);

    // Register kernel tag, platform, and metadata
    smax->register_kernel("solve_perm_Lx=b", SMAX::KernelType::SPTRSV);

    // Tell SMAX to expect a permuted matrix
    // This enables level-set scheduling for SpTRSV
    smax->kernel("solve_perm_Lx=b")->set_mat_perm(true);

    // Register operands to this kernel tag
    smax->kernel("solve_perm_Lx=b")
        ->register_A(L_n_rows, L_n_cols, L_nnz, L_col, L_row_ptr, L_val);

    // x and b are dense vectors
    smax->kernel("solve_perm_Lx=b")->register_B(A_n_rows, x_perm);
    smax->kernel("solve_perm_Lx=b")->register_C(A_n_cols, b_perm);

    // Execute all phases of this kernel
    smax->kernel("solve_perm_Lx=b")->run();

    // Unpermute solution vector
    smax->utils->apply_vec_perm<double>(A_n_cols, x_perm, x, inv_perm);

    smax->utils->print_timers();

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
