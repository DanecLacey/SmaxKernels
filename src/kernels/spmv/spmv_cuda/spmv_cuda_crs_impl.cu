#include "spmv_cuda_crs_impl.cuh"

namespace SMAX::KERNELS::SPMV::CUDA {

template <typename IT, typename VT>
__global__ void
naive_crs_spmv_cuda(const ULL n_rows, const IT *SMAX_RESTRICT col,
                    const IT *SMAX_RESTRICT row_ptr,
                    const VT *SMAX_RESTRICT val, const VT *SMAX_RESTRICT x,
                    VT *SMAX_RESTRICT y) {

    ULL row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows) {
        VT sum = VT{0};

        for (IT j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            sum += val[j] * x[col[j]];
        }

        y[row] = sum;
    }
}

template <typename IT, typename VT>
void naive_crs_spmv_cuda_launcher(const ULL n_rows, const IT *SMAX_RESTRICT col,
                                  const IT *SMAX_RESTRICT row_ptr,
                                  const VT *SMAX_RESTRICT val,
                                  const VT *SMAX_RESTRICT x,
                                  VT *SMAX_RESTRICT y) {

    // CUDA_TPB selected at compile time
    ULL blocks = (n_rows + CUDA_TPB - 1) / CUDA_TPB;

    // clang-format off
    naive_crs_spmv_cuda<IT, VT><<<blocks, CUDA_TPB>>>(n_rows, col, row_ptr, val, x, y);
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

// // Macro for cuda kernel instantiation
// #define INSTANTIATE_CRS_SPMV_KERNEL(IndexType, ValueType) \
//     template __global__ void naive_crs_spmv_cuda<IndexType, ValueType>( \
//         const ULL, const IndexType *, const IndexType *, const ValueType *, \
//         const ValueType *, ValueType *);

// // Macro for launcher instantiation
// #define INSTANTIATE_CRS_SPMV_LAUNCHER(IndexType, ValueType) \
//     template void naive_crs_spmv_cuda_launcher<IndexType, ValueType>( \
//         const ULL, const IndexType *, const IndexType *, const ValueType *, \
//         const ValueType *, ValueType *);

// // Master macro to instantiate both
// #define INSTANTIATE_CRS_SPMV(IndexType, ValueType) \
//     INSTANTIATE_CRS_SPMV_KERNEL(IndexType, ValueType); \
//     INSTANTIATE_CRS_SPMV_LAUNCHER(IndexType, ValueType);

// #define INSTANTIATE_CRS_SPMV_FLOAT_DOUBLE(IndexType)                           \
//     INSTANTIATE_CRS_SPMV(IndexType, float);                                    \
//     INSTANTIATE_CRS_SPMV(IndexType, double);

// INSTANTIATE_CRS_SPMV_FLOAT_DOUBLE(int16_t);
// INSTANTIATE_CRS_SPMV_FLOAT_DOUBLE(int32_t);
// INSTANTIATE_CRS_SPMV_FLOAT_DOUBLE(int64_t);
// INSTANTIATE_CRS_SPMV_FLOAT_DOUBLE(uint16_t);
// INSTANTIATE_CRS_SPMV_FLOAT_DOUBLE(uint32_t);
// INSTANTIATE_CRS_SPMV_FLOAT_DOUBLE(uint64_t);

// clang-format off
template __global__ void naive_crs_spmv_cuda<int16_t, float>(ULL, const int16_t*, const int16_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<int16_t, double>(ULL, const int16_t*, const int16_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<int32_t, float>(ULL, const int32_t*, const int32_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<int32_t, double>(ULL, const int32_t*, const int32_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<int64_t, float>(ULL, const int64_t*, const int64_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<int64_t, double>(ULL, const int64_t*, const int64_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<uint16_t, float>(ULL, const uint16_t*, const uint16_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<uint16_t, double>(ULL, const uint16_t*, const uint16_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<uint32_t, float>(ULL, const uint32_t*, const uint32_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<uint32_t, double>(ULL, const uint32_t*, const uint32_t*, const double*, const double*, double*);
template __global__ void naive_crs_spmv_cuda<uint64_t, float>(ULL, const uint64_t*, const uint64_t*, const float*, const float*, float*);
template __global__ void naive_crs_spmv_cuda<uint64_t, double>(ULL, const uint64_t*, const uint64_t*, const double*, const double*, double*);

template void naive_crs_spmv_cuda_launcher<int16_t, float>(ULL, const int16_t*, const int16_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<int16_t, double>(ULL, const int16_t*, const int16_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<int32_t, float>(ULL, const int32_t*, const int32_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<int32_t, double>(ULL, const int32_t*, const int32_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<int64_t, float>(ULL, const int64_t*, const int64_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<int64_t, double>(ULL, const int64_t*, const int64_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<uint16_t, float>(ULL, const uint16_t*, const uint16_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<uint16_t, double>(ULL, const uint16_t*, const uint16_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<uint32_t, float>(ULL, const uint32_t*, const uint32_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<uint32_t, double>(ULL, const uint32_t*, const uint32_t*, const double*, const double*, double*);
template void naive_crs_spmv_cuda_launcher<uint64_t, float>(ULL, const uint64_t*, const uint64_t*, const float*, const float*, float*);
template void naive_crs_spmv_cuda_launcher<uint64_t, double>(ULL, const uint64_t*, const uint64_t*, const double*, const double*, double*);
// clang-format on

} // namespace SMAX::KERNELS::SPMV::CUDA
