#pragma once

#include "../../../platforms/gpu/gpu_manager.hpp"

#include <stdint.h>
#include <stdio.h>

namespace SMAX::KERNELS::SPMV::CUDA {

template <typename IT, typename VT>
__global__ void
naive_scs_spmv_cuda(const ULL C, const ULL n_chunks,
                    const IT *SMAX_RESTRICT chunk_ptr,
                    const IT *SMAX_RESTRICT chunk_lengths,
                    const IT *SMAX_RESTRICT col, const VT *SMAX_RESTRICT val,
                    const VT *SMAX_RESTRICT x, VT *SMAX_RESTRICT y);

template <typename IT, typename VT>
void naive_scs_spmv_cuda_launcher(const ULL C, const ULL n_chunks,
                                  const IT *SMAX_RESTRICT chunk_ptr,
                                  const IT *SMAX_RESTRICT chunk_lengths,
                                  const IT *SMAX_RESTRICT col,
                                  const VT *SMAX_RESTRICT val,
                                  const VT *SMAX_RESTRICT x,
                                  VT *SMAX_RESTRICT y);

} // namespace SMAX::KERNELS::SPMV::CUDA
