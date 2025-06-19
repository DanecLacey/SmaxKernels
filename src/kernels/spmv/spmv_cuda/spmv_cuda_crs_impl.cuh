#pragma once

#include "../../../platforms/cuda/cuda.cuh"

#include <stdint.h>
#include <stdio.h>

namespace SMAX::KERNELS::SPMV::CUDA {
template <typename IT, typename VT>
__global__ void naive_crs_spmv_cuda(const ULL n_rows, const IT *RESTRICT col,
                                    const IT *RESTRICT row_ptr,
                                    const VT *RESTRICT val,
                                    const VT *RESTRICT x, VT *RESTRICT y);

template <typename IT, typename VT>
void naive_crs_spmv_cuda_launcher(const ULL n_rows, const IT *RESTRICT col,
                                  const IT *RESTRICT row_ptr,
                                  const VT *RESTRICT val, const VT *RESTRICT x,
                                  VT *RESTRICT y);

} // namespace SMAX::KERNELS::SPMV::CUDA
