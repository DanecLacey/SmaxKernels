#ifndef SMAX_SPMM_CPU_HPP
#define SMAX_SPMM_CPU_HPP

#include "../../common.hpp"
#include "spmm_cpu/spmm_cpu_core.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMM {

// These templated structs are just little helpers to wrap the core functions.
// The operator() function is called in the dispatch_spmm function to execute
// the correct function for the given types.
template <typename IT, typename VT> struct Init {
    int operator()(KernelContext context, Args *args, Flags *flags,
                   int A_offset, int X_offset, int Y_offset) {
        return SPMM_CPU::spmm_initialize_cpu_core<IT, VT>(
            context, args, flags, A_offset, X_offset, Y_offset);
    }
};

template <typename IT, typename VT> struct Apply {
    int operator()(KernelContext context, Args *args, Flags *flags,
                   int A_offset, int X_offset, int Y_offset) {
        return SPMM_CPU::spmm_apply_cpu_core<IT, VT>(
            context, args, flags, A_offset, X_offset, Y_offset);
    }
};

template <typename IT, typename VT> struct Finalize {
    int operator()(KernelContext context, Args *args, Flags *flags,
                   int A_offset, int X_offset, int Y_offset) {
        return SPMM_CPU::spmm_finalize_cpu_core<IT, VT>(
            context, args, flags, A_offset, X_offset, Y_offset);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int spmm_dispatch_cpu(KernelContext context, Args *args, Flags *flags,
                      int A_offset, int X_offset, int Y_offset) {
    switch (context.float_type) {
    case FLOAT32:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, float>()(context, args, flags, A_offset,
                                           X_offset, Y_offset);
        case UINT32:
            return Func<uint32_t, float>()(context, args, flags, A_offset,
                                           X_offset, Y_offset);
        case UINT64:
            return Func<uint64_t, float>()(context, args, flags, A_offset,
                                           X_offset, Y_offset);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FLOAT64:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, double>()(context, args, flags, A_offset,
                                            X_offset, Y_offset);
        case UINT32:
            return Func<uint32_t, double>()(context, args, flags, A_offset,
                                            X_offset, Y_offset);
        case UINT64:
            return Func<uint64_t, double>()(context, args, flags, A_offset,
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
int spmm_initialize_cpu(KernelContext context, Args *args, Flags *flags,
                        int A_offset, int X_offset, int Y_offset) {
    return spmm_dispatch_cpu<Init>(context, args, flags, A_offset, X_offset,
                                   Y_offset);
}
int spmm_apply_cpu(KernelContext context, Args *args, Flags *flags,
                   int A_offset, int X_offset, int Y_offset) {
    return spmm_dispatch_cpu<Apply>(context, args, flags, A_offset, X_offset,
                                    Y_offset);
}
int spmm_finalize_cpu(KernelContext context, Args *args, Flags *flags,
                      int A_offset, int X_offset, int Y_offset) {
    return spmm_dispatch_cpu<Finalize>(context, args, flags, A_offset, X_offset,
                                       Y_offset);
}

} // namespace SPMM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPMM_CPU_HPP
