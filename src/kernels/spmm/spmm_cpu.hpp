#pragma once

#include "../../common.hpp"
#include "spmm_cpu/spmm_cpu_core.hpp"

namespace SMAX::KERNELS::SPMM {

// These templated structs are just little helpers to wrap the core functions.
// The operator() function is called in the dispatch_spmm function to execute
// the correct function for the given types.
template <typename IT, typename VT> struct Init {
    int operator()(KernelContext *k_ctx, Args *args, Flags *flags, int A_offset,
                   int X_offset, int Y_offset) {
        return SPMM_CPU::initialize_cpu_core<IT, VT>(
            k_ctx, args, flags, A_offset, X_offset, Y_offset);
    }
};

template <typename IT, typename VT> struct Apply {
    int operator()(KernelContext *k_ctx, Args *args, Flags *flags, int A_offset,
                   int X_offset, int Y_offset) {
        return SPMM_CPU::apply_cpu_core<IT, VT>(k_ctx, args, flags, A_offset,
                                                X_offset, Y_offset);
    }
};

template <typename IT, typename VT> struct Finalize {
    int operator()(KernelContext *k_ctx, Args *args, Flags *flags, int A_offset,
                   int X_offset, int Y_offset) {
        return SPMM_CPU::finalize_cpu_core<IT, VT>(k_ctx, args, flags, A_offset,
                                                   X_offset, Y_offset);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int dispatch_cpu(KernelContext *k_ctx, Args *args, Flags *flags, int A_offset,
                 int X_offset, int Y_offset) {
    switch (k_ctx->float_type) {
    case FloatType::FLOAT32:
        switch (k_ctx->int_type) {
        case IntType::UINT16:
            return Func<uint16_t, float>()(k_ctx, args, flags, A_offset,
                                           X_offset, Y_offset);
        case IntType::UINT32:
            return Func<uint32_t, float>()(k_ctx, args, flags, A_offset,
                                           X_offset, Y_offset);
        case IntType::UINT64:
            return Func<uint64_t, float>()(k_ctx, args, flags, A_offset,
                                           X_offset, Y_offset);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FloatType::FLOAT64:
        switch (k_ctx->int_type) {
        case IntType::UINT16:
            return Func<uint16_t, double>()(k_ctx, args, flags, A_offset,
                                            X_offset, Y_offset);
        case IntType::UINT32:
            return Func<uint32_t, double>()(k_ctx, args, flags, A_offset,
                                            X_offset, Y_offset);
        case IntType::UINT64:
            return Func<uint64_t, double>()(k_ctx, args, flags, A_offset,
                                            X_offset, Y_offset);
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
int initialize_cpu(KernelContext *k_ctx, Args *args, Flags *flags, int A_offset,
                   int X_offset, int Y_offset) {
    return dispatch_cpu<Init>(k_ctx, args, flags, A_offset, X_offset, Y_offset);
}
int apply_cpu(KernelContext *k_ctx, Args *args, Flags *flags, int A_offset,
              int X_offset, int Y_offset) {
    return dispatch_cpu<Apply>(k_ctx, args, flags, A_offset, X_offset,
                               Y_offset);
}
int finalize_cpu(KernelContext *k_ctx, Args *args, Flags *flags, int A_offset,
                 int X_offset, int Y_offset) {
    return dispatch_cpu<Finalize>(k_ctx, args, flags, A_offset, X_offset,
                                  Y_offset);
}

} // namespace SMAX::KERNELS::SPMM
