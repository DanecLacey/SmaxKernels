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
    CRSMatrix<IT, VT> *A = new CRSMatrix<IT, VT>;
    A->n_rows = 8;
    A->n_cols = 8;
    A->nnz = 15;
    A->col = new IT[A->nnz]{0, 1, 0, 2, 1, 2, 3, 2, 4, 3, 5, 4, 6, 5, 7};
    A->row_ptr = new IT[A->n_rows + 1]{0, 1, 2, 4, 7, 9, 11, 13, 15};
    A->val = new VT[A->nnz]{11.0, 22.0, 31.0, 33.0, 42.0, 43.0, 44.0, 53.0,
                            55.0, 64.0, 66.0, 75.0, 77.0, 86.0, 88.0};

    DenseMatrix<VT> *x = new DenseMatrix<VT>(A->n_cols, 1, 1.0);
    DenseMatrix<VT> *b = new DenseMatrix<VT>(A->n_rows, 1, 2.0);

    // Declare permuted data
    CRSMatrix<IT, VT> *A_perm =
        new CRSMatrix<IT, VT>(A->n_rows, A->n_cols, A->nnz);
    DenseMatrix<VT> *x_perm = new DenseMatrix<VT>(A->n_cols, 1, 0.0);
    DenseMatrix<VT> *b_perm = new DenseMatrix<VT>(A->n_rows, 1, 0.0);

    // Declare permutation vectors
    int *perm = new int[A->n_rows];
    int *inv_perm = new int[A->n_rows];

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();

    // Get BFS permutation vector
    smax->utils->generate_perm<int>(A->n_rows, A->row_ptr, A->col, perm,
                                    inv_perm, std::string("BFS"));

    printf("BFS Permutation:\n");
    print_vector<int>(perm, A->n_rows);

    printf("A:\n");
    A->print();

    // Apply permutation vector to A
    smax->utils->apply_mat_perm<IT, VT>(A->n_rows, A->row_ptr, A->col, A->val,
                                        A_perm->row_ptr, A_perm->col,
                                        A_perm->val, perm, inv_perm);

    printf("A_perm:\n");
    A_perm->print();

    // Apply permutation vector to x and b
    smax->utils->apply_vec_perm<VT>(A->n_cols, x->val, x_perm->val, perm);
    smax->utils->apply_vec_perm<VT>(A->n_rows, b->val, b_perm->val, perm);

    // Declare L and U data
    CRSMatrix<IT, VT> *D_plus_L = new CRSMatrix<IT, VT>;
    CRSMatrix<IT, VT> *U = new CRSMatrix<IT, VT>;
    extract_D_L_U<IT, VT>(*A_perm, *D_plus_L, *U);

    // Register kernel tag, platform, and metadata
    smax->register_kernel("solve_perm_Lx=b", SMAX::KernelType::SPTRSV);

    // Tell SMAX to expect a permuted matrix
    // This enables level-set scheduling for SpTRSV
    smax->kernel("solve_perm_Lx=b")->set_mat_perm(true);

    // Register operands to this kernel tag
    smax->kernel("solve_perm_Lx=b")
        ->register_A(D_plus_L->n_rows, D_plus_L->n_cols, D_plus_L->nnz,
                     D_plus_L->col, D_plus_L->row_ptr, D_plus_L->val);

    // x and b are dense vectors
    smax->kernel("solve_perm_Lx=b")->register_B(D_plus_L->n_rows, x_perm->val);
    smax->kernel("solve_perm_Lx=b")->register_C(D_plus_L->n_cols, b_perm->val);

    // Execute all phases of this kernel
    smax->kernel("solve_perm_Lx=b")->run();

    // Unpermute solution vector
    smax->utils->apply_vec_perm<VT>(D_plus_L->n_cols, x_perm->val, x->val,
                                    inv_perm);

    smax->utils->print_timers();

    x->print();

    delete A;
    delete A_perm;
    delete D_plus_L;
    delete U;
    delete[] perm;
    delete[] inv_perm;
    delete x;
    delete b;
    delete x_perm;
    delete b_perm;
    delete smax;
    delete L;
    delete U;

    return 0;
}
