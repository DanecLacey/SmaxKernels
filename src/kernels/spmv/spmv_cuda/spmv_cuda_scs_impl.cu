#include "spmv_cuda_scs_impl.cuh"

namespace SMAX::KERNELS::SPMV::CUDA {

template <typename IT, typename VT>
__global__ void naive_scs_spmv_cuda(const ULL C, const ULL n_chunks,
                                    const IT *RESTRICT chunk_ptr,
                                    const IT *RESTRICT chunk_lengths,
                                    const IT *RESTRICT col,
                                    const VT *RESTRICT val,
                                    const VT *RESTRICT x, VT *RESTRICT y) {

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
                                  const IT *RESTRICT chunk_ptr,
                                  const IT *RESTRICT chunk_lengths,
                                  const IT *RESTRICT col,
                                  const VT *RESTRICT val, const VT *RESTRICT x,
                                  VT *RESTRICT y) {

    // CUDA_TPB selected at compile time
    ULL blocks = (C * n_chunks + CUDA_TPB - 1) / CUDA_TPB;

    // clang-format off
    naive_scs_spmv_cuda<IT, VT><<<blocks, CUDA_TPB>>>(C, n_chunks, chunk_ptr, chunk_lengths, col, val, x, y);
    // clang-format on

    // Synchronize device to ensure kernel execution completes
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in naive_scs_spmv_cuda_launcher: %s\n",
                cudaGetErrorString(err));
        std::exit(EXIT_FAILURE); // or throw an exception depending on your
                                 // error model
    }
}

// Macro for cuda kernel instantiation
#define INSTANTIATE_SCS_SPMV_KERNEL(IT, VT)                                    \
    template __global__ void naive_scs_spmv_cuda(                              \
        const ULL, const ULL, const IT *RESTRICT, const IT *RESTRICT,          \
        const IT *RESTRICT, const VT *RESTRICT, const VT *RESTRICT,            \
        VT *RESTRICT);

// Macro for launcher instantiation
#define INSTANTIATE_SCS_SPMV_LAUNCHER(IT, VT)                                  \
    template void naive_scs_spmv_cuda_launcher(                                \
        const ULL, const ULL, const IT *RESTRICT, const IT *RESTRICT,          \
        const IT *RESTRICT, const VT *RESTRICT, const VT *RESTRICT,            \
        VT *RESTRICT)

// Master macro to instantiate both
#define INSTANTIATE_SCS_SPMV(IndexType, ValueType)                             \
    INSTANTIATE_SCS_SPMV_KERNEL(IndexType, ValueType);                         \
    INSTANTIATE_SCS_SPMV_LAUNCHER(IndexType, ValueType);

#define INSTANTIATE_SCS_SPMV_FLOAT_DOUBLE(IndexType)                           \
    INSTANTIATE_SCS_SPMV(IndexType, float);                                    \
    INSTANTIATE_SCS_SPMV(IndexType, double);

INSTANTIATE_SCS_SPMV_FLOAT_DOUBLE(int16_t);
INSTANTIATE_SCS_SPMV_FLOAT_DOUBLE(int32_t);
INSTANTIATE_SCS_SPMV_FLOAT_DOUBLE(int64_t);
INSTANTIATE_SCS_SPMV_FLOAT_DOUBLE(uint16_t);
INSTANTIATE_SCS_SPMV_FLOAT_DOUBLE(uint32_t);
INSTANTIATE_SCS_SPMV_FLOAT_DOUBLE(uint64_t);

} // namespace SMAX::KERNELS::SPMV::CUDA
