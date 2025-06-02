#pragma once

#include "../common.hpp"
#include "utils_common.hpp"

namespace SMAX {

typedef struct {
    int index;
    int count;
} SellCSigmaPair;

template <typename IT, typename VT>
int Utils::convert_coo_to_scs(int A_coo_n_rows, int A_coo_n_cols, int A_coo_nnz,
                              IT *A_coo_col, IT *A_coo_row, VT *A_coo_val,
                              int A_scs_C, int A_scs_sigma, int A_scs_n_rows,
                              int A_scs_n_rows_padded, int A_scs_n_cols,
                              int A_scs_n_chunks, int A_scs_n_elements,
                              int A_scs_nnz, IT *A_scs_chunk_ptrs,
                              IT *A_scs_chunk_lengths, IT *A_scs_col,
                              VT *A_scs_val, IT *A_scs_perm,
                              IT *A_scs_inv_perm) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering convert_coo_to_scs"));

    // TODO

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting convert_coo_to_scs"));
    return 0;
};

} // namespace SMAX