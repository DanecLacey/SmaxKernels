#pragma once

#include "../common.hpp"
#include "utils_common.hpp"

namespace SMAX {

typedef struct {
    int index;
    int count;
} SellCSigmaPair;

static inline int compareDescSCS(const void *a, const void *b) {

    const SellCSigmaPair *pa = (const SellCSigmaPair *)a;
    const SellCSigmaPair *pb = (const SellCSigmaPair *)b;

    if (pa->count < pb->count)
        return 1; // Descending order
    if (pa->count > pb->count)
        return -1;
    return 0; // Stable if equal
}

template <typename IT, typename VT>
int Utils::convert_crs_to_scs(const int A_crs_n_rows, const int A_crs_n_cols,
                              const int A_crs_nnz, const IT *A_crs_col,
                              const IT *A_crs_row_ptr, const VT *A_crs_val,
                              const int A_scs_C, const int A_scs_sigma,
                              int &A_scs_n_rows, int &A_scs_n_rows_padded,
                              int &A_scs_n_cols, int &A_scs_n_chunks,
                              int &A_scs_n_elements, int &A_scs_nnz,
                              IT *&A_scs_chunk_ptr, IT *&A_scs_chunk_lengths,
                              IT *&A_scs_col, VT *&A_scs_val, IT *&A_scs_perm,
                              IT *&A_scs_inv_perm) {

    UtilsErrorHandler::not_implemented("convert_crs_to_scs");

    // TODO

    IF_SMAX_DEBUG(ErrorHandler::log("Entering convert_crs_to_scs"));
    const int C = A_scs_C;
    const int sigma = A_scs_sigma;
    A_scs_n_rows = A_crs_n_rows;
    A_scs_n_cols = A_crs_n_cols;
    A_scs_nnz = A_crs_nnz;
    A_scs_n_chunks = (A_crs_n_rows + C - 1) / sigma;
    A_scs_n_rows_padded = A_scs_n_chunks * C;

    // (Temporary vector) Assign an index to each row to use for row sorting
    std::vector<SellCSigmaPair> elems_per_row(A_scs_n_rows_padded);

    for (int i = 0; i < A_scs_n_rows_padded; ++i) {
        elems_per_row[i].index = i;
        elems_per_row[i].count = 0;
    }

    // Collect the number of elements in each row
    for (int i = 0; i < A_crs_n_rows; i++) {
        elems_per_row[i].count = A_crs_row_ptr[i + 1] - A_crs_row_ptr[i];
    }

    for (int i = 0; i < A_scs_n_rows_padded; i += sigma) {
        int chunk_start = i;
        int chunk_stop = std::min(i + sigma, A_scs_n_rows_padded);

        auto begin = elems_per_row.begin() + chunk_start;
        auto end = elems_per_row.begin() + chunk_stop;

        std::sort(begin, end,
                  [](const SellCSigmaPair &lhs, const SellCSigmaPair &rhs) {
                      // compareDescSCS returns <0 if lhs should come before rhs
                      return compareDescSCS(&lhs, &rhs) < 0;
                  });
    }

    A_scs_chunk_lengths = new IT[A_scs_n_chunks];
    A_scs_chunk_ptr = new IT[A_scs_n_chunks + 1];

    IT current_chunk_ptr = 0;

    for (int chunk = 0; chunk < A_scs_n_chunks; ++chunk) {
        const int chunk_offset = chunk * C;

        // Determine how many rows are actually in this chunk,
        // capping at A_scs_n_rows_padded.
        const int chunk_size =
            std::min<int>(C, A_scs_n_rows_padded - chunk_offset);

        // Find the maximum “count” field among all SellCSigmaPair in this
        // chunk.
        auto range_begin = elems_per_row.begin() + chunk_offset;
        auto range_end = range_begin + chunk_size;

        const auto max_it = std::max_element(
            range_begin, range_end, [](auto const &a, auto const &b) {
                return a.count < b.count; // compare by “count”
            });

        int max_length = 0;
        if (max_it != range_end) {
            max_length = max_it->count;
        }

        // Store chunk length and pointer offset
        A_scs_chunk_lengths[chunk] = (IT)max_length;
        A_scs_chunk_ptr[chunk] = (IT)current_chunk_ptr;

        // Advance the pointer by (chunk length × C)
        current_chunk_ptr += (max_length * C);
    }

    // Account for final chunk
    A_scs_n_elements = A_scs_chunk_ptr[A_scs_n_chunks - 1] +
                       A_scs_chunk_ptr[A_scs_n_chunks - 1] * C;

    A_scs_chunk_ptr[A_scs_n_chunks] = (IT)A_scs_n_elements;

    // Construct permutation vector
    A_scs_perm = new IT[A_scs_n_rows];
    for (int i = 0; i < A_scs_n_rows_padded; ++i) {
        IT old_row = elems_per_row[i].index;
        if (old_row < A_scs_n_rows)
            A_scs_perm[old_row] = (IT)i;
    }

    // Construct inverse permutation vector
    A_scs_inv_perm = new IT[A_scs_n_rows];
    for (int i = 0; i < A_scs_n_rows; ++i) {
#ifdef DEBUG_MODE
        // Sanity check for common error
        // TODO: Wrap in ErrorHandler
        if (A_scs_inv_perm[i] >= A_scs_n_rows) {
            fprintf(stderr,
                    "ERROR convert_crs_to_scs: A_scs_inv_perm[%d]=%d"
                    " is out of bounds (>%d).\n",
                    i, A_scs_inv_perm[i], A_scs_n_rows);
        }
#endif
        A_scs_inv_perm[A_scs_perm[i]] = (IT)i;
    }

    // Now that chunk data is collected, fill with matrix data
    A_scs_col = new IT[A_scs_n_elements];
    A_scs_val = new VT[A_scs_n_elements];

    // Initialize defaults (essential for padded elements)
    for (int i = 0; i < A_scs_n_elements; ++i) {
        A_scs_val[i] = (VT)0.0;
        A_scs_col[i] = (IT)0;
        // TODO: may need to offset when used with MPI
        // A_scs_col[i] = padded_val;
    }

    // (Temporary vector) Keep track of how many elements we've seen in each row
    std::vector<int> row_local_elem_count(A_scs_n_rows_padded, 0);

    for (int i = 0; i < A_scs_n_rows; ++i) {
        int old_row = i;
        for (int j = A_crs_row_ptr[i]; j < A_crs_row_ptr[i + 1]; ++j) {
            int new_row = A_scs_perm[old_row];
            int chunk_idx = new_row / C;
            int chunk_start = A_scs_chunk_ptr[chunk_idx];
            int chunk_row = new_row % C;
            int idx =
                chunk_start + row_local_elem_count[new_row] * C + chunk_row;
            A_scs_col[idx] = A_crs_col[j];
            A_scs_val[idx] = A_crs_val[j];
            ++row_local_elem_count[new_row];
#ifdef DEBUG_MODE
            // Sanity check for common error
            // TODO: Wrap in ErrorHandler
            if (A_scs_col[idx] >= A_scs_n_cols) {
                fprintf(stderr,
                        "ERROR convert_crs_to_scs: A_scs_col[%d]=%d"
                        " is out of bounds (>%d).\n",
                        idx, A_scs_col[idx], A_scs_n_cols);
            }
#endif
        }
    }

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting convert_crs_to_scs"));
    return 0;
};

} // namespace SMAX