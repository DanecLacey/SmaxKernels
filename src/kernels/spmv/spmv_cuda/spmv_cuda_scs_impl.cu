#include "spmv_cuda_scs_impl.cuh"

namespace SMAX::KERNELS::SPMV::CUDA {

template <typename IT, typename VT>
__global__ void
naive_scs_spmv_cuda(const ULL C, const ULL n_chunks,
                    const IT *SMAX_RESTRICT chunk_ptr,
                    const IT *SMAX_RESTRICT chunk_lengths,
                    const IT *SMAX_RESTRICT col, const VT *SMAX_RESTRICT val,
                    const VT *SMAX_RESTRICT x, VT *SMAX_RESTRICT y) {

    long row = threadIdx.x + blockDim.x * blockIdx.x;
    ULL c = row / C;   // the no. of the chunk
    ULL idx = row % C; // index inside the chunk

    if (row < n_chunks * C) {
        VT tmp{};
        IT cs = chunk_ptr[c];

        for (IT j = 0; j < chunk_lengths[c]; ++j) {
            ULL offset = cs + j * C + idx;
            tmp += val[offset] * x[col[offset]];
        }

        y[row] = tmp;
    }
}

template <typename IT, typename VT>
void naive_scs_spmv_cuda_launcher(const ULL C, const ULL n_chunks,
                                  const IT *SMAX_RESTRICT chunk_ptr,
                                  const IT *SMAX_RESTRICT chunk_lengths,
                                  const IT *SMAX_RESTRICT col,
                                  const VT *SMAX_RESTRICT val,
                                  const VT *SMAX_RESTRICT x,
                                  VT *SMAX_RESTRICT y) {

    // CUDA_TPB selected at compile time
    ULL blocks = (C * n_chunks + CUDA_TPB - 1) / CUDA_TPB;

    // clang-format off
    naive_scs_spmv_cuda<IT, VT><<<blocks, CUDA_TPB>>>(C, n_chunks, chunk_ptr, chunk_lengths, col, val, x, y);
    // clang-format on

    // Synchronize device to ensure kernel execution completes
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in naive_crs_spmv_cuda_launcher: %s\n",
                cudaGetErrorString(err));
        std::exit(EXIT_FAILURE); // or throw an exception depending on your
                                 // error model
    }
}

// Explicit instantiations
// clang-format off
template __global__ void naive_scs_spmv_cuda<int16_t, float>(const ULL, const ULL, const int16_t*, const int16_t*, const int16_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<int16_t, double>(const ULL, const ULL, const int16_t*, const int16_t*, const int16_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<int32_t, float>(const ULL, const ULL, const int32_t*, const int32_t*, const int32_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<int32_t, double>(const ULL, const ULL, const int32_t*, const int32_t*, const int32_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<int64_t, float>(const ULL, const ULL, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<int64_t, double>(const ULL, const ULL, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<uint16_t, float>(const ULL, const ULL, const uint16_t*, const uint16_t*, const uint16_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<uint16_t, double>(const ULL, const ULL, const uint16_t*, const uint16_t*, const uint16_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<uint32_t, float>(const ULL, const ULL, const uint32_t*, const uint32_t*, const uint32_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<uint32_t, double>(const ULL, const ULL, const uint32_t*, const uint32_t*, const uint32_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<uint64_t, float>(const ULL, const ULL, const uint64_t*, const uint64_t*, const uint64_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<uint64_t, double>(const ULL, const ULL, const uint64_t*, const uint64_t*, const uint64_t*, const double*, const double*, double*);

template void naive_scs_spmv_cuda_launcher<int16_t, float>(const ULL, const ULL, const int16_t*, const int16_t*, const int16_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<int16_t, double>(const ULL, const ULL, const int16_t*, const int16_t*, const int16_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<int32_t, float>(const ULL, const ULL, const int32_t*, const int32_t*, const int32_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<int32_t, double>(const ULL, const ULL, const int32_t*, const int32_t*, const int32_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<int64_t, float>(const ULL, const ULL, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<int64_t, double>(const ULL, const ULL, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<uint16_t, float>(const ULL, const ULL, const uint16_t*, const uint16_t*, const uint16_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<uint16_t, double>(const ULL, const ULL, const uint16_t*, const uint16_t*, const uint16_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<uint32_t, float>(const ULL, const ULL, const uint32_t*, const uint32_t*, const uint32_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<uint32_t, double>(const ULL, const ULL, const uint32_t*, const uint32_t*, const uint32_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<uint64_t, float>(const ULL, const ULL, const uint64_t*, const uint64_t*, const uint64_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<uint64_t, double>(const ULL, const ULL, const uint64_t*, const uint64_t*, const uint64_t*, const double*, const double*, double*);
// clang-format on

} // namespace SMAX::KERNELS::SPMV::CUDA
