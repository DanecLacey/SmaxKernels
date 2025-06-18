#pragma once
// clang-format off
#include "memory_utils.hpp"
#include "common.hpp"
#include "error_handler.hpp"
#include "kernel.hpp"
#include <string>
// clang-format on

namespace SMAX {

class Utils {
  private:
    std::unordered_map<std::string, std::unique_ptr<Kernel>> &kernels;

  public:
    UtilitiesContainer *uc = nullptr;
    Utils(std::unordered_map<std::string, std::unique_ptr<Kernel>> &_kernels,
          UtilitiesContainer *_uc)
        : kernels(_kernels), uc(_uc) {}

    ~Utils() {}

    void print_timers();

    template <typename IT, typename VT>
    int convert_crs_to_scs(const int _n_rows, const int _n_cols, const int _nnz,
                           const IT *_col, const IT *_row_ptr, const VT *_val,
                           const int C, const int sigma, int &n_rows,
                           int &n_rows_padded, int &n_cols, int &n_chunks,
                           int &n_elements, int &nnz, IT *&chunk_ptr,
                           IT *&chunk_lengths, IT *&col, VT *&val, IT *&perm);

    template <typename IT>
    int build_symmetric_csr(IT *A_row_ptr, IT *A_col, int A_n_rows,
                            IT *&A_sym_row_ptr, IT *&A_sym_col, int &A_sym_nnz);

    template <typename IT>
    int generate_perm_jh(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                         int *inv_perm, int *lvl);

    template <typename IT>
    int generate_perm_DFS(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                          int *inv_perm, int *lvl);

    template <typename IT>
    int generate_perm_BFS(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                          int *inv_perm, int *lvl);

    template <typename IT>
    int generate_color_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                            int *inv_perm, int *lvl);

    template <typename IT>
    void generate_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, int *perm,
                       int *inv_perm, std::string type = std::string("BFS"));

    template <typename IT, typename VT>
    void apply_mat_perm(int A_n_rows, IT *A_row_ptr, IT *A_col, VT *A_val,
                        IT *A_perm_row_ptr, IT *A_perm_col, VT *A_perm_val,
                        int *perm, int *inv_perm);

    template <typename VT>
    void apply_vec_perm(int n_rows, VT *vec, VT *vec_perm, int *perm);
};

} // namespace SMAX

// DL 06.05.2025 NOTE: Don't love forward declaring the class.. Works for now
#include "utils/convert_formats.hpp"
#include "utils/permutations.hpp"
#include "utils/timers.hpp"
