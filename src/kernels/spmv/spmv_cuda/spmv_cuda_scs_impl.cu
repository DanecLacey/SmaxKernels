#include "spmv_cuda_scs_impl.cuh"

namespace SMAX::KERNELS::SPMV::SPMV_CUDA {

template <typename IT, typename VT>
__global__ void naive_scs_spmv_cuda(const int C, const int n_chunks,
                                    const IT *RESTRICT chunk_ptr,
                                    const IT *RESTRICT chunk_lengths,
                                    const IT *RESTRICT col,
                                    const VT *RESTRICT val,
                                    const VT *RESTRICT x, VT *RESTRICT y) {

    long row = threadIdx.x + blockDim.x * blockIdx.x;
    int c = row / C;   // the no. of the chunk
    int idx = row % C; // index inside the chunk

    if (row < n_chunks * C) {
        VT tmp{};
        int cs = chunk_ptr[c];

        for (int j = 0; j < chunk_lengths[c]; ++j) {
            int offset = cs + j * C + idx;
            tmp += values[offset] * x[col_idxs[offset]];
        }

        y[row] = tmp;
    }
}

template <typename IT, typename VT>
void naive_scs_spmv_cuda_launcher(const int C, const int n_chunks,
                                  const IT *RESTRICT chunk_ptr,
                                  const IT *RESTRICT chunk_lengths,
                                  const IT *RESTRICT col,
                                  const VT *RESTRICT val, const VT *RESTRICT x,
                                  VT *RESTRICT y) {

    // CUDA_TPB selected at compile time
    int blocks = (A_n_rows + CUDA_TPB - 1) / CUDA_TPB;

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
template __global__ void naive_scs_spmv_cuda<int16_t, float>(const int, const int, const int16_t*, const int16_t*, const int16_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<int16_t, double>(const int, const int, const int16_t*, const int16_t*, const int16_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<int32_t, float>(const int, const int, const int32_t*, const int32_t*, const int32_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<int32_t, double>(const int, const int, const int32_t*, const int32_t*, const int32_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<int64_t, float>(const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<int64_t, double>(const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<uint16_t, float>(const int, const int, const uint16_t*, const uint16_t*, const uint16_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<uint16_t, double>(const int, const int, const uint16_t*, const uint16_t*, const uint16_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<uint32_t, float>(const int, const int, const uint32_t*, const uint32_t*, const uint32_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<uint32_t, double>(const int, const int, const uint32_t*, const uint32_t*, const uint32_t*, const double*, const double*, double*);
template __global__ void naive_scs_spmv_cuda<uint64_t, float>(const int, const int, const uint64_t*, const uint64_t*, const uint64_t*, const float*, const float*, float*);
template __global__ void naive_scs_spmv_cuda<uint64_t, double>(const int, const int, const uint64_t*, const uint64_t*, const uint64_t*, const double*, const double*, double*);

template void naive_scs_spmv_cuda_launcher<int16_t, float>(const int, const int, const int16_t*, const int16_t*, const int16_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<int16_t, double>(const int, const int, const int16_t*, const int16_t*, const int16_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<int32_t, float>(const int, const int, const int32_t*, const int32_t*, const int32_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<int32_t, double>(const int, const int, const int32_t*, const int32_t*, const int32_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<int64_t, float>(const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<int64_t, double>(const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<uint16_t, float>(const int, const int, const uint16_t*, const uint16_t*, const uint16_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<uint16_t, double>(const int, const int, const uint16_t*, const uint16_t*, const uint16_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<uint32_t, float>(const int, const int, const uint32_t*, const uint32_t*, const uint32_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<uint32_t, double>(const int, const int, const uint32_t*, const uint32_t*, const uint32_t*, const double*, const double*, double*);
template void naive_scs_spmv_cuda_launcher<uint64_t, float>(const int, const int, const uint64_t*, const uint64_t*, const uint64_t*, const float*, const float*, float*);
template void naive_scs_spmv_cuda_launcher<uint64_t, double>(const int, const int, const uint64_t*, const uint64_t*, const uint64_t*, const double*, const double*, double*);
// clang-format on

} // namespace SMAX::KERNELS::SPMV::SPMV_CUDA
