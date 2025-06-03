#pragma once

#include "../../../platforms/cuda/cuda.cuh"

#include <stdint.h>
#include <stdio.h>

namespace SMAX::KERNELS::SPMV::SPMV_CUDA {
template <typename IT, typename VT>
__global__ void
naive_crs_spmv_cuda(const int A_n_rows, const IT *RESTRICT A_col,
                    const IT *RESTRICT A_row_ptr, const VT *RESTRICT A_val,
                    const VT *RESTRICT x, VT *RESTRICT y);

template <typename IT, typename VT>
void naive_crs_spmv_cuda_launcher(const int A_n_rows, const IT *RESTRICT A_col,
                                  const IT *RESTRICT A_row_ptr,
                                  const VT *RESTRICT A_val,
                                  const VT *RESTRICT x, VT *RESTRICT y);

} // namespace SMAX::KERNELS::SPMV::SPMV_CUDA
