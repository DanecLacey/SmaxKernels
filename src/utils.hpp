#pragma once

#include "common.hpp"
#include "error_handler.hpp"

namespace SMAX {

class Utils {
  private:
    UtilitiesContainer *uc = nullptr;

  public:
    Utils(UtilitiesContainer *_uc) : uc(_uc) {}

    ~Utils() {}

    template <typename IT>
    void generate_perm_jh(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                          int *inv_perm);

    template <typename IT>
    void generate_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                       int *inv_perm);

    template <typename IT, typename VT>
    void apply_mat_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, VT *A_val,
                        IT *A_perm_row_ptr, IT *A_perm_col, VT *A_perm_val,
                        int *perm, int *inv_perm);

    template <typename VT>
    void apply_vec_perm(int n_rows, VT *vec, VT *vec_perm, int *perm);
};

} // namespace SMAX

// DL 06.05.2025 NOTE: Don't love forward declaring the class.. Works for now
#include "utils/permutation_impl.hpp"
#include "utils/timers.hpp"
