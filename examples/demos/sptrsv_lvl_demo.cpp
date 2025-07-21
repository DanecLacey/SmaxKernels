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

int main(void) {

    using IT = int;
    using VT = double;
    using ULL = unsigned long long int;

    // Initialize operands
    ULL A_n_rows = 8;
    ULL A_n_cols = 8;
    ULL A_nnz = 15;
    IT *A_col = new IT[A_nnz]{0, 1, 0, 2, 1, 2, 3, 2, 4, 3, 5, 4, 6, 5, 7};
    IT *A_row_ptr = new IT[A_n_rows + 1]{0, 1, 2, 4, 7, 9, 11, 13, 15};
    VT *A_val = new VT[A_nnz]{11.0, 22.0, 31.0, 33.0, 42.0, 43.0, 44.0, 53.0,
                              55.0, 64.0, 66.0, 75.0, 77.0, 86.0, 88.0};

    VT *x = new VT[A_n_cols];
    for (ULL i = 0; i < A_n_cols; ++i) {
        x[i] = 1.0;
    }

    // Initialize RHS
    VT *b = new VT[A_n_rows];
    for (ULL i = 0; i < A_n_rows; ++i) {
        b[i] = 2.0;
    }

    // Declare permuted data
    CRSMatrix<IT, VT> *A_perm = new CRSMatrix<IT, VT>(A_n_rows, A_n_cols, A_nnz);
    VT *x_perm = new VT[A_n_cols];
    VT *b_perm = new VT[A_n_rows];

    // Declare permutation vectors
    int *perm = new int[A_n_rows];
    int *inv_perm = new int[A_n_rows];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Get BFS permutation vector
    smax->utils->generate_perm<int>(A_n_rows, A_row_ptr, A_col, perm, inv_perm,
                                    std::string("BFS"));

    printf("BFS Permutation:\n");
    print_vector<int>(perm, A_n_rows);

    printf("A:\n");
    print_matrix(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val);

    // Apply permutation vector to A
    smax->utils->apply_mat_perm<IT, VT>(A_n_rows, A_row_ptr, A_col, A_val,
                                        A_perm->row_ptr, A_perm->col, A_perm->val,
                                        perm, inv_perm);

    printf("A_perm:\n");
    print_matrix<IT, VT>(A_n_rows, A_n_cols, A_nnz, A_perm->col, A_perm->row_ptr,
                         A_perm->val);

    // Apply permutation vector to x and b
    smax->utils->apply_vec_perm<VT>(A_n_cols, x, x_perm, perm);
    smax->utils->apply_vec_perm<VT>(A_n_rows, b, b_perm, perm);

    // Declare L and U data
    CRSMatrix<IT, VT> *L = new CRSMatrix<IT, VT>();
    CRSMatrix<IT, VT> *U = new CRSMatrix<IT, VT>();

    extract_D_L_U<IT, VT>(*A_perm, *L, *U);

    // Register kernel tag, platform, and metadata
    smax->register_kernel("solve_perm_Lx=b", SMAX::KernelType::SPTRSV);

    // Tell SMAX to expect a permuted matrix
    // This enables level-set scheduling for SpTRSV
    smax->kernel("solve_perm_Lx=b")->set_mat_perm(true);

    // Register operands to this kernel tag
    smax->kernel("solve_perm_Lx=b")
        ->register_A(L->n_rows, L->n_cols, L->nnz, L->col, L->row_ptr, L->val);

    // x and b are dense vectors
    smax->kernel("solve_perm_Lx=b")->register_B(A_n_rows, x_perm);
    smax->kernel("solve_perm_Lx=b")->register_C(A_n_cols, b_perm);

    // Execute all phases of this kernel
    smax->kernel("solve_perm_Lx=b")->run();

    // Unpermute solution vector
    smax->utils->apply_vec_perm<VT>(A_n_cols, x_perm, x, inv_perm);

    smax->utils->print_timers();

    print_vector<VT>(x, A_n_cols);

    delete[] A_col;
    delete[] A_row_ptr;
    delete[] A_val;
    delete A_perm;
    delete[] perm;
    delete[] inv_perm;
    delete[] x;
    delete[] b;
    delete[] x_perm;
    delete[] b_perm;
    delete smax;
    delete L;
    delete U;

    return 0;
}
