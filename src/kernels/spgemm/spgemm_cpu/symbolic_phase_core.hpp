#ifndef SMAX_SYMBOLIC_PHASE_HPP
#define SMAX_SYMBOLIC_PHASE_HPP

#include "../spgemm_common.hpp"
#include "symbolic_phase_impl.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {
namespace SPGEMM_CPU {

template <typename IT, typename VT>
int symbolic_phase_cpu(KernelContext context, Args *args, Flags *flags) {
    IF_DEBUG(ErrorHandler::log("Entering symbolic_phase_cpu"));

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = args->A->n_rows;
    int A_n_cols = args->A->n_cols;
    int A_nnz = args->A->nnz;
    IT *A_col = as<IT *>(args->A->col);
    IT *A_row_ptr = as<IT *>(args->A->row_ptr);
    VT *A_val = as<VT *>(args->A->val);

    int B_n_rows = args->B->n_rows;
    int B_n_cols = args->B->n_cols;
    int B_nnz = args->B->nnz;
    IT *B_col = as<IT *>(args->B->col);
    IT *B_row_ptr = as<IT *>(args->B->row_ptr);
    VT *B_val = as<VT *>(args->B->val);

    // Since we want to reallocate the data pointed to by _C,
    // we need references to each of the pointers
    int &C_n_rows = *(args->C->n_rows);
    int &C_n_cols = *(args->C->n_cols);
    int &C_nnz = *(args->C->nnz);
    IT *&C_col = as<IT *>(args->C->col);
    IT *&C_row_ptr = as<IT *>(args->C->row_ptr);
    VT *&C_val = as<VT *>(args->C->val);

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

#endif // SMAX_SYMBOLIC_PHASE_HPP
