#pragma once

#include "../../common.hpp"
#if USE_CUDA
#include "spmv_cuda/spmv_cuda_core.cuh"
#endif

namespace SMAX::KERNELS::SPMV {

// These templated structs are just little helpers to wrap the core functions.
// The operator() function is called in the dispatch_spmv function to execute
// the correct function for the given types.
template <typename IT, typename VT> struct Init_CUDA {
    int operator()(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags, int A_offset, int x_offset, int y_offset) {
#if USE_CUDA
        return SPMV_CUDA::initialize_cuda_core<IT, VT>(
            timers, k_ctx, args, flags, A_offset, x_offset, y_offset);
#else
        return 1; // TODO: Handle error properly
#endif
    }
};

template <typename IT, typename VT> struct Apply_CUDA {
    int operator()(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags, int A_offset, int x_offset, int y_offset) {
#if USE_CUDA
        return SPMV_CUDA::apply_cuda_core<IT, VT>(timers, k_ctx, args, flags,
                                                  A_offset, x_offset, y_offset);
#else
        return 1; // TODO: Handle error properly
#endif
    }
};

template <typename IT, typename VT> struct Finalize_CUDA {
    int operator()(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags, int A_offset, int x_offset, int y_offset) {
#if USE_CUDA
        return SPMV_CUDA::finalize_cuda_core<IT, VT>(
            timers, k_ctx, args, flags, A_offset, x_offset, y_offset);
#else
        return 1; // TODO: Handle error properly
#endif
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
// Dispatch kernel based on data types
template <template <typename IT, typename VT> class Func>
int dispatch_cuda(Timers *timers, KernelContext *k_ctx, Args *args,
                  Flags *flags, int A_offset, int x_offset, int y_offset) {
    switch (k_ctx->float_type) {
    case FloatType::FLOAT32:
        switch (k_ctx->int_type) {
        case IntType::INT16:
            return Func<int16_t, float>()(timers, k_ctx, args, flags, A_offset,
                                          x_offset, y_offset);
        case IntType::INT32:
            return Func<int32_t, float>()(timers, k_ctx, args, flags, A_offset,
                                          x_offset, y_offset);
        case IntType::INT64:
            return Func<int64_t, float>()(timers, k_ctx, args, flags, A_offset,
                                          x_offset, y_offset);
        case IntType::UINT16:
            return Func<uint16_t, float>()(timers, k_ctx, args, flags, A_offset,
                                           x_offset, y_offset);
        case IntType::UINT32:
            return Func<uint32_t, float>()(timers, k_ctx, args, flags, A_offset,
                                           x_offset, y_offset);
        case IntType::UINT64:
            return Func<uint64_t, float>()(timers, k_ctx, args, flags, A_offset,
                                           x_offset, y_offset);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FloatType::FLOAT64:
        switch (k_ctx->int_type) {
        case IntType::INT16:
            return Func<int16_t, double>()(timers, k_ctx, args, flags, A_offset,
                                           x_offset, y_offset);
        case IntType::INT32:
            return Func<int32_t, double>()(timers, k_ctx, args, flags, A_offset,
                                           x_offset, y_offset);
        case IntType::INT64:
            return Func<int64_t, double>()(timers, k_ctx, args, flags, A_offset,
                                           x_offset, y_offset);
        case IntType::UINT16:
            return Func<uint16_t, double>()(timers, k_ctx, args, flags,
                                            A_offset, x_offset, y_offset);
        case IntType::UINT32:
            return Func<uint32_t, double>()(timers, k_ctx, args, flags,
                                            A_offset, x_offset, y_offset);
        case IntType::UINT64:
            return Func<uint64_t, double>()(timers, k_ctx, args, flags,
                                            A_offset, x_offset, y_offset);
        default:
            std::cerr << "Unsupported int type\n";
            return 1;
        }
    default:
        std::cerr << "Error: Float type not supported\n";
        return 1;
    }

    return 0;
}

// These invoke the dispatcher function with the correct template parameters
int initialize_cuda(Timers *timers, KernelContext *k_ctx, Args *args,
                    Flags *flags, int A_offset, int x_offset, int y_offset) {
#if USE_CUDA
    return dispatch_cuda<Init_CUDA>(timers, k_ctx, args, flags, A_offset,
                                    x_offset, y_offset);
#else
    return 1; // TODO: Handle error properly
#endif
}
int apply_cuda(Timers *timers, KernelContext *k_ctx, Args *args, Flags *flags,
               int A_offset, int x_offset, int y_offset) {
#if USE_CUDA
    return dispatch_cuda<Apply_CUDA>(timers, k_ctx, args, flags, A_offset,
                                     x_offset, y_offset);
#else
    return 1; // TODO: Handle error properly
#endif
}
int finalize_cuda(Timers *timers, KernelContext *k_ctx, Args *args,
                  Flags *flags, int A_offset, int x_offset, int y_offset) {
#if USE_CUDA
    return dispatch_cuda<Finalize_CUDA>(timers, k_ctx, args, flags, A_offset,
                                        x_offset, y_offset);
#else
    return 1; // TODO: Handle error properly
#endif
}

} // namespace SMAX::KERNELS::SPMV
