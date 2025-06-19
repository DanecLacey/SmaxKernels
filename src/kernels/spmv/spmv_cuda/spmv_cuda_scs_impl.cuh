#pragma once

#include "../../../platforms/cuda/cuda.cuh"

#include <stdint.h>
#include <stdio.h>

namespace SMAX::KERNELS::SPMV::CUDA {

template <typename IT, typename VT>
__global__ void naive_scs_spmv_cuda(const ULL C, const ULL n_chunks,
                                    const IT *RESTRICT chunk_ptr,
                                    const IT *RESTRICT chunk_lengths,
                                    const IT *RESTRICT col,
                                    const VT *RESTRICT val,
                                    const VT *RESTRICT x, VT *RESTRICT y);

template <typename IT, typename VT>
void naive_scs_spmv_cuda_launcher(const ULL C, const ULL n_chunks,
                                  const IT *RESTRICT chunk_ptr,
                                  const IT *RESTRICT chunk_lengths,
                                  const IT *RESTRICT col,
                                  const VT *RESTRICT val, const VT *RESTRICT x,
                                  VT *RESTRICT y);

} // namespace SMAX::KERNELS::SPMV::CUDA
