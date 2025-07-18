#pragma once

#include "../../../platforms/gpu/gpu_manager.hpp"

#include <stdint.h>
#include <stdio.h>

namespace SMAX::KERNELS::SPMV::CUDA {
template <typename IT, typename VT>
__global__ void
naive_crs_spmv_cuda(const ULL n_rows, const IT *SMAX_RESTRICT col,
                    const IT *SMAX_RESTRICT row_ptr,
                    const VT *SMAX_RESTRICT val, const VT *SMAX_RESTRICT x,
                    VT *SMAX_RESTRICT y);

template <typename IT, typename VT>
void naive_crs_spmv_cuda_launcher(const ULL n_rows, const IT *SMAX_RESTRICT col,
                                  const IT *SMAX_RESTRICT row_ptr,
                                  const VT *SMAX_RESTRICT val,
                                  const VT *SMAX_RESTRICT x,
                                  VT *SMAX_RESTRICT y);

} // namespace SMAX::KERNELS::SPMV::CUDA
