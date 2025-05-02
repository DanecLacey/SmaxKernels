#ifndef SYMBOLIC_PHASE_HPP
#define SYMBOLIC_PHASE_HPP

#include "../spgemm_common.hpp"
#include "symbolic_phase_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {
namespace SPGEMM_CPU {

template <typename IT, typename VT>
int symbolic_phase_cpu(KernelContext context, SparseMatrix *_A,
                       SparseMatrix *_B, SparseMatrix *_C) {
    IF_DEBUG(ErrorHandler::log("Entering symbolic_phase_cpu"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    IT A_n_rows = as<IT>(_A->n_rows);
    IT A_n_cols = as<IT>(_A->n_cols);
    IT A_nnz = as<IT>(_A->nnz);
    IT *A_col = as<IT *>(_A->col);
    IT *A_row_ptr = as<IT *>(_A->row_ptr);
    VT *A_val = as<VT *>(_A->val);

    IT B_n_rows = as<IT>(_B->n_rows);
    IT B_n_cols = as<IT>(_B->n_cols);
    IT B_nnz = as<IT>(_B->nnz);
    IT *B_col = as<IT *>(_B->col);
    IT *B_row_ptr = as<IT *>(_B->row_ptr);
    VT *B_val = as<VT *>(_B->val);

    // Since we want to reallocate the data pointed to by _C,
    // we need references to each of the pointers
    IT &C_n_rows = as<IT>(_C->n_rows);
    IT &C_n_cols = as<IT>(_C->n_cols);
    IT &C_nnz = as<IT>(_C->nnz);
    IT *&C_col = as<IT *>(_C->col);
    IT *&C_row_ptr = as<IT *>(_C->row_ptr);
    VT *&C_val = as<VT *>(_C->val);

#if 1
    padded_symbolic_phase(A_n_rows, A_n_cols, A_nnz, A_col, A_row_ptr, A_val,
                          B_n_rows, B_n_cols, B_nnz, B_col, B_row_ptr, B_val,
                          C_n_rows, C_n_cols, C_nnz, C_col, C_row_ptr, C_val);
#endif

    IF_DEBUG(ErrorHandler::log("Exiting symbolic_phase_cpu"));
    return 0;
}

} // namespace SPGEMM_CPU
} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // SYMBOLIC_PHASE_HPP
