#include "spmv_cuda_crs_impl.cuh"

namespace SMAX::KERNELS::SPMV::SPMV_CUDA {

template <typename IT, typename VT>
__global__ void naive_crs_spmv_cuda(int A_n_rows, const IT *RESTRICT A_col,
                                    const IT *RESTRICT A_row_ptr,
                                    const VT *RESTRICT A_val,
                                    const VT *RESTRICT x, VT *RESTRICT y) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A_n_rows) {
        VT sum = VT{0};

        for (IT j = A_row_ptr[row]; j < A_row_ptr[row + 1]; ++j) {
            IT col = A_col[j];
            sum += A_val[j] * x[col];
        }
        y[row] = sum;
    }
}

template <typename IT, typename VT>
void naive_crs_spmv_cuda_launcher(int A_n_rows, const IT *RESTRICT A_col,
                                  const IT *RESTRICT A_row_ptr,
                                  const VT *RESTRICT A_val,
                                  const VT *RESTRICT x, VT *RESTRICT y) {

    // CUDA_TPB selected at compile time
    int blocks = (A_n_rows + CUDA_TPB - 1) / CUDA_TPB;

    // clang-format off
    naive_crs_spmv_cuda<IT, VT><<<blocks, CUDA_TPB>>>(A_n_rows, A_col, A_row_ptr, A_val, x, y);
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

#define INSTANTIATE_SPMV_KERNEL(IndexType, ValueType)                          \
    template __global__ void naive_crs_spmv_cuda<IndexType, ValueType>(        \
        int, const IndexType *, const IndexType *, const ValueType *,          \
        const ValueType *, ValueType *);

// Macro for launcher instantiation
#define INSTANTIATE_SPMV_LAUNCHER(IndexType, ValueType)                        \
    template void naive_crs_spmv_cuda_launcher<IndexType, ValueType>(          \
        int, const IndexType *, const IndexType *, const ValueType *,          \
        const ValueType *, ValueType *);

// Master macro to instantiate both
#define INSTANTIATE_SPMV(IndexType, ValueType)                                 \
    INSTANTIATE_SPMV_KERNEL(IndexType, ValueType);                             \
    INSTANTIATE_SPMV_LAUNCHER(IndexType, ValueType)

// Now instantiate for all required types
INSTANTIATE_SPMV(int16_t, float);
INSTANTIATE_SPMV(int16_t, double);
INSTANTIATE_SPMV(int32_t, float);
INSTANTIATE_SPMV(int32_t, double);
INSTANTIATE_SPMV(int64_t, float);
INSTANTIATE_SPMV(int64_t, double);
INSTANTIATE_SPMV(uint16_t, float);
INSTANTIATE_SPMV(uint16_t, double);
INSTANTIATE_SPMV(uint32_t, float);
INSTANTIATE_SPMV(uint32_t, double);
INSTANTIATE_SPMV(uint64_t, float);
INSTANTIATE_SPMV(uint64_t, double);

} // namespace SMAX::KERNELS::SPMV::SPMV_CUDA
