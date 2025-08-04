#pragma once

#include "../../../platforms/cuda/cuda.cuh"

#include <stdint.h>
#include <stdio.h>

namespace SMAX::KERNELS::SPMV::CUDA {
template <typename IT, typename VT, bool block_column_major>
__global__ void naive_bcrs_spmv_cuda_thread_per_row(
    const ULL n_rows, const ULL b_height, const ULL b_width,
    const ULL height_pad, const ULL width_pad, const IT *SMAX_RESTRICT col,
    const IT *SMAX_RESTRICT row_ptr, const VT *SMAX_RESTRICT val,
    const VT *SMAX_RESTRICT x, VT *SMAX_RESTRICT y);

template <typename IT, typename VT, bool block_column_major>
__global__ void naive_bcrs_spmv_cuda_warp_per_row(
    const ULL n_rows, const ULL b_height, const ULL b_width,
    const ULL height_pad, const ULL width_pad, const IT *SMAX_RESTRICT col,
    const IT *SMAX_RESTRICT row_ptr, const VT *SMAX_RESTRICT val,
    const VT *SMAX_RESTRICT x, VT *SMAX_RESTRICT y, const ULL power_hint);

template <typename IT, typename VT>
__global__ void naive_bcrs_spmv_cuda_warp_per_row_by_shffl(
    const ULL n_rows, const ULL b_height, const ULL b_width,
    const ULL height_pad, const ULL width_pad, const IT *SMAX_RESTRICT col,
    const IT *SMAX_RESTRICT row_ptr, const VT *SMAX_RESTRICT val,
    const VT *SMAX_RESTRICT x, VT *SMAX_RESTRICT y, const ULL power_hint);

template <typename IT, typename VT>
void naive_bcrs_spmv_cuda_launcher(
    const ULL n_rows, const ULL b_height, const ULL b_width,
    const ULL height_pad, const ULL width_pad, const IT *SMAX_RESTRICT col,
    const IT *SMAX_RESTRICT row_ptr, const VT *SMAX_RESTRICT val,
    const VT *SMAX_RESTRICT x, VT *SMAX_RESTRICT y, int krn_type,
    bool block_column_major);

} // namespace SMAX::KERNELS::SPMV::CUDA