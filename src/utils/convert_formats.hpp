#pragma once

#include "../common.hpp"
#include "utils_common.hpp"
#include <algorithm>

namespace SMAX {

// More clear than using std::pair with "first" and "second"
typedef struct {
    int index;
    int count;
} SellCSigmaPair;

template <typename IT, typename VT, typename ST>
int Utils::convert_crs_to_scs(const ST _n_rows, const ST _n_cols, const ST _nnz,
                              const IT *_col, const IT *_row_ptr,
                              const VT *_val, const ST C, const ST sigma,
                              ST &n_rows, ST &n_rows_padded, ST &n_cols,
                              ST &n_chunks, ST &n_elements, ST &nnz,
                              IT *&chunk_ptr, IT *&chunk_lengths, IT *&col,
                              VT *&val, IT *&perm) {

    IF_SMAX_DEBUG(ErrorHandler::log("Entering convert_crs_to_scs"));
    n_rows = _n_rows;
    n_cols = _n_cols;
    nnz = _nnz;
    n_chunks = (_n_rows + C - 1) / C;
    n_rows_padded = n_chunks * C;

    // (Temporary vector) Assign an index to each row to use for row sorting
    std::vector<SellCSigmaPair> elems_per_row(n_rows_padded);

    for (ST i = 0; i < n_rows_padded; ++i) {
        elems_per_row[i].index = i;
        elems_per_row[i].count = 0;
    }

    // Collect the number of elements in each row
    // NOTE: All rows from n_rows -> n_rows_padded are 0 count
    for (ST i = 0; i < n_rows; i++) {
        elems_per_row[i].count = _row_ptr[i + 1] - _row_ptr[i];
    }

    for (ST i = 0; i < n_rows_padded; i += sigma) {
        auto begin = &elems_per_row[i];
        auto end = (i + sigma) < n_rows_padded ? &elems_per_row[i + sigma]
                                               : &elems_per_row[n_rows_padded];

        std::sort(
            begin, end,
            // sort longer rows first
            [](const auto &a, const auto &b) { return a.count > b.count; });
    }

    chunk_lengths = new IT[n_chunks];
    chunk_ptr = new IT[n_chunks + 1];

    IT current_chunk_ptr = 0;

    for (ST chunk = 0; chunk < n_chunks; ++chunk) {
        auto begin = &elems_per_row[chunk * C];
        auto end = &elems_per_row[chunk * C + C];

        chunk_lengths[chunk] =
            std::max_element(begin, end, [](const auto &a, const auto &b) {
                return a.count < b.count;
            })->count;

        chunk_ptr[chunk] = current_chunk_ptr;
        current_chunk_ptr += chunk_lengths[chunk] * C;
    }

    // Account for final chunk
    n_elements = chunk_ptr[n_chunks - 1] + chunk_lengths[n_chunks - 1] * C;

    chunk_ptr[n_chunks] = (IT)n_elements;

    // // Construct permutation vector
    perm = new IT[n_rows];
    for (int i = 0; i < n_rows_padded; ++i) {
        IT old_row = elems_per_row[i].index;
        if (old_row < n_rows)
            perm[old_row] = (IT)i;
    }

    // Now that chunk data is collected, fill with matrix data
    col = new IT[n_elements];
    val = new VT[n_elements];

    // Initialize defaults (essential for padded elements)
    for (ST i = 0; i < n_elements; ++i) {
        val[i] = VT{};
        col[i] = IT{};
        // TODO: may need to offset when used with MPI
        // col[i] = padded_val;
    }

    // (Temporary vector) Keep track of how many elements we've seen in each row
    std::vector<ST> row_local_elem_count(n_rows_padded, 0);

    for (ST i = 0; i < n_rows; ++i) {
        int old_row = i;
        for (int j = _row_ptr[i]; j < _row_ptr[i + 1]; ++j) {
            ST new_row = perm[old_row];
            ST chunk_idx = new_row / C;
            ST chunk_start = chunk_ptr[chunk_idx];
            ST chunk_row = new_row % C;
            ST idx =
                chunk_start + row_local_elem_count[new_row] * C + chunk_row;
            col[idx] = _col[j];
            val[idx] = _val[j];
            ++row_local_elem_count[new_row];

            // Common errors
#ifdef DEBUG_MODE
            if (col[idx] >= n_cols) {
                UtilsErrorHandler::col_ob(col[idx], idx, n_cols,
                                          std::string("convert_crs_to_scs"));
            }
            if (col[idx] < 0) {
                UtilsErrorHandler::col_ub(col[idx], idx, n_cols,
                                          std::string("convert_crs_to_scs"));
            }
#endif
        }
    }

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting convert_crs_to_scs"));
    return 0;
};

} // namespace SMAX
