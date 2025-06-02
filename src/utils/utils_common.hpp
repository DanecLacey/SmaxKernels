#pragma once

#include "../common.hpp"

namespace SMAX {

class UtilsErrorHandler : public ErrorHandler {
  public:
    static void utils_fatal(const std::string &message) {
        fatal("[UtilsError] " + message);
    }

    static void utils_warning(const std::string &message) {
        warning("[UtilsWarning] " + message);
    }

    template <typename IT>
    static void col_ob(IT col_value, int j, int max_cols,
                       const std::string &util_name) {
        std::ostringstream oss;
        oss << "Column index " << col_value << " at position " << j
            << " is out of bounds (max = " << max_cols - 1 << ").";
        utils_fatal("[" + util_name + "] " + oss.str());
    }

    template <typename IT>
    static void col_ub(IT col_value, int j, int min_cols,
                       const std::string &util_name) {
        std::ostringstream oss;
        oss << "Column index " << col_value << " at position " << j
            << " is out of bounds (min = " << min_cols - 1 << ").";
        utils_fatal("[" + util_name + "] " + oss.str());
    }
    static void perm_type_dne(const std::string &failed_type,
                              const char *available_types) {
        std::ostringstream oss;
        oss << "Permutation type: " << failed_type << " does not exist.\n";
        oss << "Please choose a type in: " << available_types << ".\n";
        utils_fatal(oss.str());
    }

    static void not_implemented(const std::string &util_name) {
        std::ostringstream oss;
        oss << "This utility is not yet implemented.";
        utils_fatal("[" + util_name + "] " + oss.str());
    }
};

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
