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

// Explicit instantiations
// clang-format off
template __global__ void naive_crs_spmv_cuda<int16_t, float>(int, const int16_t*, const int16_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<int16_t, double>(int, const int16_t*, const int16_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<int32_t, float>(int, const int32_t*, const int32_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<int32_t, double>(int, const int32_t*, const int32_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<int64_t, float>(int, const int64_t*, const int64_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<int64_t, double>(int, const int64_t*, const int64_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<uint16_t, float>(int, const uint16_t*, const uint16_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<uint16_t, double>(int, const uint16_t*, const uint16_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<uint32_t, float>(int, const uint32_t*, const uint32_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<uint32_t, double>(int, const uint32_t*, const uint32_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<uint64_t, float>(int, const uint64_t*, const uint64_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<uint64_t, double>(int, const uint64_t*, const uint64_t*, const double*, const double*, double*);

template void naive_crs_spmv_cuda_launcher<int16_t, float>(int, const int16_t*, const int16_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<int16_t, double>(int, const int16_t*, const int16_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<int32_t, float>(int, const int32_t*, const int32_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<int32_t, double>(int, const int32_t*, const int32_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<int64_t, float>(int, const int64_t*, const int64_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<int64_t, double>(int, const int64_t*, const int64_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<uint16_t, float>(int, const uint16_t*, const uint16_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<uint16_t, double>(int, const uint16_t*, const uint16_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<uint32_t, float>(int, const uint32_t*, const uint32_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<uint32_t, double>(int, const uint32_t*, const uint32_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<uint64_t, float>(int, const uint64_t*, const uint64_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<uint64_t, double>(int, const uint64_t*, const uint64_t*, const double*, const double*, double*);
// clang-format on

} // namespace SMAX::KERNELS::SPMV::SPMV_CUDA
