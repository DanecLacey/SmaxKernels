#pragma once

#include "../common.hpp"
#include "utils_common.hpp"
#include <algorithm>
#include <set>

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
#ifdef SMAX_DEBUG_MODE
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

template <typename IT, typename VT, typename ST>
int Utils::convert_crs_to_bcrs(const ST _n_rows, const ST _n_cols, const ST _nnz,
                              const IT *_col, const IT *_row_ptr,
                              const VT *_val, ST& n_rows, ST& n_cols, ST& nnz, ST& b_height, ST& b_width,
                              ST& height_pad, ST& width_pad, IT *&col, IT *&row_ptr, VT *&val,
                              const ST target_b_height, const ST target_b_width,
                              const ST target_height_pad, const ST target_width_pad, const bool block_column_major) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering convert_crs_to_bcrs"));
    // check if size match
    if(_n_rows%target_b_height)
    {
        throw std::runtime_error("Target block height does not divide number of input rows");
    }
    if(_n_cols%target_b_width)
    {
        throw std::runtime_error("Target block width does not divide number of input columns");
    }
    if(target_height_pad < target_b_height)
    {
        throw std::runtime_error("Height padding has to be at least target block height");
    }
    if(target_height_pad < target_b_height)
    {
        throw std::runtime_error("Width padding has to be at least target block width");
    }

    // setup sizes
    n_rows = _n_rows / target_b_height;
    n_cols = _n_cols / target_b_width;
    b_height = target_b_height;
    b_width = target_b_width;
    height_pad = target_height_pad;
    width_pad = target_width_pad;

    // temprorary arrays for row and column ptr
    std::vector<IT> tmp_row_ptr(n_rows+1u);
    std::vector<IT> tmp_col_ptr;

    // reserve to sensical size, we will have at most nnz block entries
    tmp_col_ptr.reserve(_nnz);

    // for each block row, we gather our values in a sorted set
    std::set<IT> tmp_uni_col;

    tmp_row_ptr[0] = IT(0);
    // we now run through block height rows and gather the columns to be added
    for(ST b_row = 0; b_row < n_rows; ++b_row)
    {
        tmp_uni_col.clear();
        for(ST l_row = b_row*b_height; l_row < (b_row+1)*b_height; ++l_row)
        {
            for(IT idx = _row_ptr[l_row]; idx < _row_ptr[l_row+1]; ++idx)
            {
                tmp_uni_col.insert(_col[idx]/b_width);
            }
        }
        tmp_row_ptr[b_row+1] = tmp_uni_col.size() + tmp_row_ptr[b_row];
        std::for_each(tmp_uni_col.begin(), tmp_uni_col.end(), [&](const auto& ele){tmp_col_ptr.push_back(ele);});
    }

    // clear set
    tmp_uni_col.clear();

    // we define nnz as non zero blocks, which is the size of our columns array
    nnz = tmp_col_ptr.size();

    // create actual matrix data
    row_ptr = new IT[n_rows+1u];
    col = new IT[nnz];
    // copy known data
    std::copy(tmp_row_ptr.begin(), tmp_row_ptr.end(), row_ptr);
    std::copy(tmp_col_ptr.begin(), tmp_col_ptr.end(), col);

    // clear used data
    tmp_row_ptr.clear();
    tmp_col_ptr.clear();

    // need space for nnz blocks
    val = new VT[nnz*height_pad*width_pad];

    // init to zero
    std::fill(val, val + nnz*height_pad*width_pad, VT(0));

    // and now run through our csr matrix blockwise, track the column position and copy into the blocked matrix
    for(ST b_row = 0; b_row < n_rows; ++b_row)
    {
        for(ST l_row = ST(0); l_row < ST(b_height); ++l_row)
        {
            ST b_idx = row_ptr[b_row];
            for(IT idx = _row_ptr[b_row*b_height + l_row]; idx < _row_ptr[b_row*b_height + l_row + 1]; ++idx)
            {
                // advance block row index until it is the same
                while(col[b_idx] < _col[idx]/b_width)
                {
                    ++b_idx;
                }
                // todo: only in debug modus?
                if(col[b_idx] != _col[idx]/b_width)
                {
                    throw std::runtime_error("Blocked Column array does not fit with csr columns");
                }
                // and now write the value into the correct block idx
                const IT blc_inc = block_column_major ? (height_pad * (_col[idx]%b_width) + l_row) : (width_pad * l_row + (_col[idx]%b_width));
                // std::printf("b_row %i, b_idx %i, l_row %i, idx %i, blc_inc %i\n", int(b_row), int(b_idx), int(l_row), int(idx), int(blc_inc));
                val[b_idx*height_pad*width_pad + blc_inc] = _val[idx];
            }
        }
    }

    IF_SMAX_DEBUG(ErrorHandler::log("Exiting convert_crs_to_bcrs"));
    return 0;
};

} // namespace SMAX
