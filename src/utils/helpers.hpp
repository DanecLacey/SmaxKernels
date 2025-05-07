#ifndef SMAX_UTIL_HELPERS_HPP
#define SMAX_UTIL_HELPERS_HPP

#include <queue>

namespace SMAX {

template <typename IT>
void print_matrix(int n_rows, int n_cols, int nnz, IT *col, IT *row_ptr) {
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
    printf("\n");
}

} // namespace SMAX

#endif // SMAX_UTIL_HELPERS_HPP