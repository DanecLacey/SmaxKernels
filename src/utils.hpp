#ifndef SMAX_UTILS_HPP
#define SMAX_UTILS_HPP

namespace SMAX {

class Utils {
  public:
    // Constructors (if needed)
    Utils() = default;

    template <typename IT>
    void generate_perm_jh(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                       int *inv_perm);

    template <typename IT>
    void generate_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                       int *inv_perm);

    template <typename IT, typename VT>
    void apply_mat_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, VT *A_val,
                        IT *A_perm_row_ptr, IT *A_perm_col, VT *A_perm_val,
                        int *perm);

    template <typename VT>
    void apply_vec_perm(int n_rows, VT *vec, VT *vec_perm, int *perm);

  private:
    // Internal helpers, if needed
};

} // namespace SMAX

// DL 06.05.2025 NOTE: Don't love forward declaring the class.. Works for now
#include "utils/permutation.hpp"

#endif // SMAX_UTILS_HPP
