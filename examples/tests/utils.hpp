#pragma once

#include <iostream>

template <typename VT> void print_vector(VT *vec, int n_rows) {
    printf("Vector: [");
    for (int i = 0; i < n_rows; ++i) {
        std::cout << vec[i] << ", ";
    }
    printf("]\n\n");
}

template <typename IT, typename VT>
void print_sparse_vector(int nnz, IT *idx, VT *vec) {
    printf("Vector: [");
    for (int i = 0; i < nnz; ++i) {
        std::cout << "idx: " << idx[i] << ", ";
        std::cout << "val: " << vec[i] << std::endl;
    }
    printf("]\n\n");
}

template <typename IT, typename VT>
void print_matrix(int n_rows, int n_cols, int nnz, IT *col, IT *row_ptr,
                  VT *val, bool symbolic = false) {
    printf("n_rows = %i\n", n_rows);
    printf("n_cols = %i\n", n_cols);
    printf("nnz = %i\n", nnz);
    printf("col = [");
    for (int i = 0; i < nnz; ++i) {
        std::cout << col[i] << ", ";
    }
    printf("]\n");

    printf("row_ptr = [");
    for (int i = 0; i <= n_rows; ++i) {
        std::cout << row_ptr[i] << ", ";
    }
    printf("]\n");

    if (!symbolic) {
        printf("val = [");
        for (int i = 0; i < nnz; ++i) {
            std::cout << val[i] << ", ";
        }
        printf("]\n");
    }
    printf("\n");
}