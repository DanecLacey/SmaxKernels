#pragma once

#include "../../../common.hpp"
#include "../../kernels_common.hpp"
#include "../spmv_common.hpp"

namespace SMAX::KERNELS::SPMV::CPU {

/**
 *  Block Layout:
 *
 *  each entry in row_ptr references a block of consecutive data in val of size
 * h_pad * w_pad
 *
 *  If block_column_major is true, the block is sorted in column major fassion,
 * i.e. data in format (a[0,0], a[1,0], ...., a[0,1], a[1,1], .....)
 *
 */
template <typename IT, typename VT, bool block_column_major>
inline void
naive_bcrs_spmv(const ULL n_rows, const ULL n_cols, const ULL block_height,
                const ULL block_width, const ULL height_padding,
                const ULL width_padding, const IT *SMAX_RESTRICT col,
                const IT *SMAX_RESTRICT row_ptr, const VT *SMAX_RESTRICT val,
                const VT *SMAX_RESTRICT x, VT *SMAX_RESTRICT y) {
    // clang-format off
    IF_SMAX_DEBUG(
      if (height_padding < block_height)
          SpMVErrorHandler::kernel_fatal("Height padding must be equal or greater than block height");
      if (width_padding < block_width)
          SpMVErrorHandler::kernel_fatal("Width padding must be equal or greater than block width");
    );
    const ULL block_size = height_padding * width_padding;
    // open the parallel region to prepare data beforehand
#pragma omp parallel
    {
        VT* y_buffer = new VT[block_height];


#pragma omp for schedule(static)
          for (ULL row = 0; row < n_rows; ++row) {
            //format buffer
            for(ULL ix = 0; ix < block_height; ++ix)
            {
              y_buffer[ix] = VT(0);
            }

            // calculate local product
            for (IT j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
                IF_SMAX_DEBUG(
                    if (col[j] < 0 || col[j] >= (IT)n_cols)
                          SpMVErrorHandler::col_oob<IT>(col[j], j, n_cols);
                  );
                  // IF_SMAX_DEBUG_3(
                  //     SpMVErrorHandler::print_bcrs_elem<IT, VT>(
                  //         val[block_size*j], col, x[col[j]], j);
                  // );
                if constexpr(block_column_major) {
                    for(ULL jx = 0; jx < block_width; ++jx) {
                        for(ULL ix = 0; ix < block_height; ++ix) {
                            y_buffer[ix] += val[j*block_size + jx*height_padding + ix] * x[col[j]*width_padding + jx];
                        }
                    }
                }
                else {
                    for(ULL ix = 0; ix < block_height; ++ix) {
                        for(ULL jx = 0; jx < block_width; ++jx) {
                            y_buffer[ix] += val[j*block_size + ix*width_padding + jx] * x[col[j]*width_padding + jx];
                        }
                    }
                }
            }

            // write out
            for(ULL ix = 0; ix < block_height; ++ix)
            {
              y[row*height_padding + ix] = y_buffer[ix];
            }
        }

        delete[] y_buffer;
    }
    IF_SMAX_DEBUG_3(printf("Finish BSpMV\n"));
    // clang-format on
}

} // namespace SMAX::KERNELS::SPMV::CPU