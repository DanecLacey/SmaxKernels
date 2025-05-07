#ifndef SMAx_SPMV_CPU_HPP
#define SMAx_SPMV_CPU_HPP

#include "../../common.hpp"
#include "spmv_cpu/spmv_cpu_core.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPMV {

// These templated structs are just little helpers to wrap the core functions.
// The operator() function is called in the dispatch_spmv function to execute
// the correct function for the given types.
template <typename IT, typename VT> struct Init {
    int operator()(KernelContext context, Args *args, Flags *flags,
                   int A_offset, int x_offset, int y_offset) {
        return SPMV_CPU::spmv_initialize_cpu_core<IT, VT>(
            context, args, flags, A_offset, x_offset, y_offset);
    }
};

template <typename IT, typename VT> struct Apply {
    int operator()(KernelContext context, Args *args, Flags *flags,
                   int A_offset, int x_offset, int y_offset) {
        return SPMV_CPU::spmv_apply_cpu_core<IT, VT>(
            context, args, flags, A_offset, x_offset, y_offset);
    }
};

template <typename IT, typename VT> struct Finalize {
    int operator()(KernelContext context, Args *args, Flags *flags,
                   int A_offset, int x_offset, int y_offset) {
        return SPMV_CPU::spmv_finalize_cpu_core<IT, VT>(
            context, args, flags, A_offset, x_offset, y_offset);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int spmv_dispatch_cpu(KernelContext context, Args *args, Flags *flags,
                      int A_offset, int x_offset, int y_offset) {
    switch (context.float_type) {
    case FLOAT32:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, float>()(context, args, flags, A_offset,
                                           x_offset, y_offset);
        case UINT32:
            return Func<uint32_t, float>()(context, args, flags, A_offset,
                                           x_offset, y_offset);
        case UINT64:
            return Func<uint64_t, float>()(context, args, flags, A_offset,
                                           x_offset, y_offset);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FLOAT64:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, double>()(context, args, flags, A_offset,
                                            x_offset, y_offset);
        case UINT32:
            return Func<uint32_t, double>()(context, args, flags, A_offset,
                                            x_offset, y_offset);
        case UINT64:
            return Func<uint64_t, double>()(context, args, flags, A_offset,
                                            x_offset, y_offset);
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
int spmv_initialize_cpu(KernelContext context, Args *args, Flags *flags,
                        int A_offset, int x_offset, int y_offset) {
    return spmv_dispatch_cpu<Init>(context, args, flags, A_offset, x_offset,
                                   y_offset);
}
int spmv_apply_cpu(KernelContext context, Args *args, Flags *flags,
                   int A_offset, int x_offset, int y_offset) {
    return spmv_dispatch_cpu<Apply>(context, args, flags, A_offset, x_offset,
                                    y_offset);
}
int spmv_finalize_cpu(KernelContext context, Args *args, Flags *flags,
                      int A_offset, int x_offset, int y_offset) {
    return spmv_dispatch_cpu<Finalize>(context, args, flags, A_offset, x_offset,
                                       y_offset);
}

} // namespace SPMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAx_SPMV_CPU_HPP
