#include "bspmv_cuda_bcrs_impl.cuh"

namespace SMAX::KERNELS::BSPMV::CUDA {

__host__ inline unsigned int next_pow_2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

// global shared buffer
extern __shared__ char buffer[];

template <typename IT, typename VT, bool block_column_major>
__global__ void
naive_bcrs_spmv_cuda_thread_per_row(const ULL n_rows, const ULL b_height, const ULL b_width, const ULL height_pad, const ULL width_pad,
                    const IT *SMAX_RESTRICT col, const IT *SMAX_RESTRICT row_ptr,
                    const VT *SMAX_RESTRICT val, const VT *SMAX_RESTRICT x,
                    VT *SMAX_RESTRICT y) {
    const ULL start_row = blockIdx.x * blockDim.x + threadIdx.x;
    // offset into our actual local block buffer
    VT* loc_y = reinterpret_cast<VT*>(&buffer[0] + threadIdx.x * b_height* sizeof(VT));

    // grid strided for loop
    for(ULL row = start_row; row < n_rows; row += blockDim.x * gridDim.x)
    {
        for(int k = 0; k < b_height; ++k)
        {
            loc_y[k] = VT(0);
        }
        for (IT j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            if constexpr(block_column_major) {
                for (int w(0) ; w < b_width ; ++w) {
                    for (int h(0) ; h < b_height ; ++h) {
                        loc_y[h] += val[j * height_pad*width_pad + w * height_pad + h] * x[col[j] * width_pad + w];
                    }
                }

            }
            else {
                for (int h(0) ; h < b_height ; ++h) {
                    for (int w(0) ; w < b_width ; ++w) {
                        loc_y[h] += val[j * height_pad*width_pad + h * width_pad + w] * x[col[j] * width_pad + w];
                    }
                }
            }
        }
        for(int k = 0; k < b_height; ++k)
        {
            y[row*height_pad + k] = loc_y[k];
        }
    }
}

template <typename IT, typename VT, bool block_column_major>
__global__ void
naive_bcrs_spmv_cuda_warp_per_row(const ULL n_rows, const ULL b_height, const ULL b_width, const ULL height_pad, const ULL width_pad,
                    const IT *SMAX_RESTRICT col, const IT *SMAX_RESTRICT row_ptr,
                    const VT *SMAX_RESTRICT val, const VT *SMAX_RESTRICT x,
                    VT *SMAX_RESTRICT y, const ULL power_hint) {
    const ULL warp_idx = ULL(threadIdx.x) / ULL(warpSize);
    const ULL thread_idx = ULL(threadIdx.x) % ULL(warpSize);
    const ULL warp_block_size = ULL(blockDim.x) / ULL(warpSize);
    const ULL start_row = ULL(blockIdx.x) * warp_block_size + warp_idx;
    const ULL bs = b_height*b_width;
    const ULL columns_per_iter = ULL(warpSize)/bs;
    // our constant col offset, -1 means the thread is deactivated
    const int m_col = (thread_idx/bs < columns_per_iter) ? int(thread_idx / bs) : int(-1);
    // local row and column for this thread
    const int l_row = block_column_major ? (thread_idx%b_height) : ((thread_idx%bs)/b_width);
    const int l_col = block_column_major ? ((thread_idx%bs)/b_height) : (thread_idx%b_width);
    // we use a dynamic shared buffer, where each thread gathers local y values
    // predetermined regarding their pre allocated block idx
    // offset into our actual local block buffer, here, always inflate to warpSize
    VT* loc_buffer = reinterpret_cast<VT*>(&buffer[warpSize*warp_idx*sizeof(VT)]);

    // grid strided for loop
    for(ULL row = start_row; row < n_rows; row += warp_block_size*ULL(gridDim.x))
    {
        // here all our threads simply set shared mem to zero
        loc_buffer[thread_idx] = VT(0);
        // loop with all active threads, where each thread handles its offset into the blocked matrix
        for (IT j = row_ptr[row] + m_col; m_col >= 0 && j < row_ptr[row + 1]; j += columns_per_iter) {
            if constexpr(block_column_major) {
                loc_buffer[m_col*bs + l_col*b_height + l_row] +=  val[j * height_pad*width_pad + l_col * height_pad + l_row] * x[col[j] * width_pad + l_col];
            }
            else {
                // write local values out in column major format, since this is easier for reduction
                loc_buffer[m_col*bs + l_col*b_height + l_row] += val[j * height_pad*width_pad + l_row * width_pad + l_col] * x[col[j] * width_pad + l_col];
            }
        }
        // now reduce to first b_height threads with a cyclic reduction and write out
        for(ULL cur_cutoff = power_hint; cur_cutoff >= ULL(1); cur_cutoff /= ULL(2))
        {
            const ULL offset = cur_cutoff*b_height;
            if(thread_idx < offset)
            {
                loc_buffer[thread_idx] += thread_idx+offset < warpSize ? loc_buffer[thread_idx+ offset] : VT(0);
            }
        }
        if(thread_idx < b_height)
            y[row*height_pad+thread_idx] = loc_buffer[thread_idx];
    }
}

template <typename IT, typename VT>
__global__ void
naive_bcrs_spmv_cuda_warp_per_row_by_shffl(const ULL n_rows, const ULL b_height, const ULL b_width, const ULL height_pad, const ULL width_pad,
                    const IT *SMAX_RESTRICT col, const IT *SMAX_RESTRICT row_ptr,
                    const VT *SMAX_RESTRICT val, const VT *SMAX_RESTRICT x,
                    VT *SMAX_RESTRICT y, const ULL power_hint) {
    const ULL warp_idx = ULL(threadIdx.x) / ULL(warpSize);
    const ULL thread_idx = ULL(threadIdx.x) % ULL(warpSize);
    const ULL warp_block_size = ULL(blockDim.x) / ULL(warpSize);
    const ULL start_row = ULL(blockIdx.x) * warp_block_size + warp_idx;
    const ULL bs = b_height*b_width;
    const ULL columns_per_iter = ULL(warpSize)/bs;
    // our constant col offset, -1 means the thread is deactivated
    const int m_col = (thread_idx/bs < columns_per_iter) ? int(thread_idx / bs) : int(-1);
    // local row and column for this thread
    const int l_row = (thread_idx%b_height);
    const int l_col = ((thread_idx%bs)/b_height);

    // reduction only in local registers
    VT loc_sum = VT(0);

    // grid strided for loop
    for(ULL row = start_row; row < n_rows; row += warp_block_size*ULL(gridDim.x))
    {
        loc_sum = VT(0);
        // loop with all active threads, where each thread handles its offset into the blocked matrix
        for (IT j = row_ptr[row] + m_col; m_col >= 0 && j < row_ptr[row + 1]; j += columns_per_iter) {
            loc_sum +=  val[j * height_pad*width_pad + l_col * height_pad + l_row] * x[col[j] * width_pad + l_col];
        }
        // now do a warp shuffle down into the first b_height threads
        for(ULL cur_cutoff = power_hint; cur_cutoff >= ULL(1); cur_cutoff /= ULL(2))
        {
            loc_sum += __shfl_down_sync(0xffff, loc_sum, int(cur_cutoff*b_height));
        }
        // now correct values are inside the first b_height threads
        if(thread_idx < b_height)
            y[row*height_pad+thread_idx] = loc_sum;
    }
}

// Specific implementations for BSPMV for BCRS matrix
enum InternalBCRSKernelType : int { naive_thread_per_row = 0 , naive_warp_group = 1 , naive_warp_shuffle = 2 };

template <typename IT, typename VT>
void naive_bcrs_spmv_cuda_launcher(const ULL n_rows, const ULL b_height, const ULL b_width,
                                  const ULL height_pad, const ULL width_pad,
                                  const IT *SMAX_RESTRICT col,
                                  const IT *SMAX_RESTRICT row_ptr,
                                  const VT *SMAX_RESTRICT val,
                                  const VT *SMAX_RESTRICT x,
                                  VT *SMAX_RESTRICT y,
                                  const int krn_type, const bool block_column_major) {

    std::cout << "Entering cuda kernel\n";
    constexpr ULL warp_size = 32;
    using uint = unsigned int;
    if (krn_type == naive_thread_per_row)
    {
        ULL blocks = (n_rows + CUDA_TPB - 1) / CUDA_TPB;
        ULL shared_mem = CUDA_TPB * b_height * sizeof(VT);

        // clang-format off
        if(block_column_major)
            naive_bcrs_spmv_cuda_thread_per_row<IT, VT, true><<<blocks, CUDA_TPB, shared_mem>>>(n_rows, b_height, b_width, height_pad, width_pad,
                                col, row_ptr, val, x, y);
        else
            naive_bcrs_spmv_cuda_thread_per_row<IT, VT, false><<<blocks, CUDA_TPB, shared_mem>>>(n_rows, b_height, b_width, height_pad, width_pad,
                                col, row_ptr, val, x, y);
        // clang-format on

    }
    else if(krn_type == naive_warp_group)
    {
        if(b_height*b_width > warp_size)
        {
            throw std::runtime_error("Warp group strategy requires b x h smaller warp size");
        }
        const ULL warps_per_block = CUDA_TPB / warp_size;
        ULL blocks = (n_rows + warps_per_block - 1) / warps_per_block;
        ULL shared_mem = warps_per_block * b_height * b_width * sizeof(VT);
        ULL power_hint = ULL(next_pow_2(uint(warp_size)/(2u*uint(b_height))));

        // clang-format off
        if(block_column_major)
            naive_bcrs_spmv_cuda_warp_per_row<IT, VT, true><<<blocks, CUDA_TPB, shared_mem>>>(n_rows, b_height, b_width, height_pad, width_pad,
                                col, row_ptr, val, x, y, power_hint);
        else
            naive_bcrs_spmv_cuda_warp_per_row<IT, VT, false><<<blocks, CUDA_TPB, shared_mem>>>(n_rows, b_height, b_width, height_pad, width_pad,
                                col, row_ptr, val, x, y, power_hint);
        // clang-format on

    }
    else if(krn_type == naive_warp_shuffle)
    {
        if (!block_column_major)
        {
            throw std::runtime_error("BCSR spmv warp shuffle only works with block column major ordering");
        }
        if(b_height*b_width > warp_size)
        {
            throw std::runtime_error("Warp group strategy requires b x h smaller warp size");
        }
        const ULL warps_per_block = CUDA_TPB / warp_size;
        ULL blocks = (n_rows + warps_per_block - 1) / warps_per_block;
        ULL shared_mem = warps_per_block * b_height * b_width * sizeof(VT);
        ULL power_hint = ULL(next_pow_2(uint(warp_size)/(2u*uint(b_height))));

        // clang-format off
        naive_bcrs_spmv_cuda_warp_per_row_by_shffl<IT, VT><<<blocks, CUDA_TPB, shared_mem>>>(n_rows, b_height, b_width, height_pad, width_pad,
                                col, row_ptr, val, x, y, power_hint);
        // clang-format on

    }
    else
    {
        throw std::runtime_error("Kernel not implemented");
    }

    // Synchronize device to ensure kernel execution completes
    // TODO: optionally shut off device synch ?
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in naive_crs_spmv_cuda_launcher: %s\n",
                cudaGetErrorString(err));
        std::exit(EXIT_FAILURE); // or throw an exception depending on your
                                 // error model
    }
}

// Macro for cuda kernel instantiation
#define INSTANTIATE_BCRS_SPMV_KERNEL(IndexType, ValueType, ColMajor)                      \
    template __global__ void                                                    \
    naive_bcrs_spmv_cuda_thread_per_row<IndexType, ValueType, ColMajor>(                  \
        const ULL, const ULL, const ULL, const ULL, const ULL,                  \
        const IndexType *SMAX_RESTRICT,                                         \
        const IndexType *SMAX_RESTRICT, const ValueType *SMAX_RESTRICT,         \
        const ValueType *SMAX_RESTRICT, ValueType *SMAX_RESTRICT);

#define INSTANTIATE_BCRS_SPMV_KERNEL_WARP_GROUP(IndexType, ValueType, ColMajor)                      \
    template __global__ void                                                    \
    naive_bcrs_spmv_cuda_warp_per_row<IndexType, ValueType, ColMajor>(                  \
        const ULL, const ULL, const ULL, const ULL, const ULL,                  \
        const IndexType *SMAX_RESTRICT,                                         \
        const IndexType *SMAX_RESTRICT, const ValueType *SMAX_RESTRICT,         \
        const ValueType *SMAX_RESTRICT, ValueType *SMAX_RESTRICT, const ULL);

#define INSTANTIATE_BCRS_SPMV_KERNEL_WARP_SHFFL(IndexType, ValueType)                      \
    template __global__ void                                                    \
    naive_bcrs_spmv_cuda_warp_per_row_by_shffl<IndexType, ValueType>(                  \
        const ULL, const ULL, const ULL, const ULL, const ULL,                  \
        const IndexType *SMAX_RESTRICT,                                         \
        const IndexType *SMAX_RESTRICT, const ValueType *SMAX_RESTRICT,         \
        const ValueType *SMAX_RESTRICT, ValueType *SMAX_RESTRICT, const ULL);

// Macro for launcher instantiation
#define INSTANTIATE_BCRS_SPMV_LAUNCHER(IndexType, ValueType)                    \
    template void naive_bcrs_spmv_cuda_launcher<IndexType, ValueType>(          \
        const ULL, const ULL, const ULL, const ULL, const ULL,                  \
        const IndexType *SMAX_RESTRICT,                                         \
        const IndexType *SMAX_RESTRICT, const ValueType *SMAX_RESTRICT,        \
        const ValueType *SMAX_RESTRICT, ValueType *SMAX_RESTRICT,              \
        const int, const bool);

// Master macro to instantiate both
#define INSTANTIATE_BCRS_SPMV(IndexType, ValueType)                             \
    INSTANTIATE_BCRS_SPMV_KERNEL(IndexType, ValueType, true);                         \
    INSTANTIATE_BCRS_SPMV_KERNEL(IndexType, ValueType, false);                         \
    INSTANTIATE_BCRS_SPMV_KERNEL_WARP_GROUP(IndexType, ValueType, true);                         \
    INSTANTIATE_BCRS_SPMV_KERNEL_WARP_GROUP(IndexType, ValueType, false);                         \
    INSTANTIATE_BCRS_SPMV_KERNEL_WARP_SHFFL(IndexType, ValueType);                         \
    INSTANTIATE_BCRS_SPMV_LAUNCHER(IndexType, ValueType);

#define INSTANTIATE_BCRS_SPMV_FLOAT_DOUBLE(IndexType)                           \
    INSTANTIATE_BCRS_SPMV(IndexType, float);                                    \
    INSTANTIATE_BCRS_SPMV(IndexType, double);

INSTANTIATE_BCRS_SPMV_FLOAT_DOUBLE(int16_t);
INSTANTIATE_BCRS_SPMV_FLOAT_DOUBLE(int32_t);
INSTANTIATE_BCRS_SPMV_FLOAT_DOUBLE(int64_t);
INSTANTIATE_BCRS_SPMV_FLOAT_DOUBLE(uint16_t);
INSTANTIATE_BCRS_SPMV_FLOAT_DOUBLE(uint32_t);
INSTANTIATE_BCRS_SPMV_FLOAT_DOUBLE(uint64_t);

} // namespace SMAX::KERNELS::BSPMV::CUDA
