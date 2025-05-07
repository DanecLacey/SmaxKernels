#ifndef SMAX_SPGEMV_CPU_HPP
#define SMAX_SPGEMV_CPU_HPP

#include "../../common.hpp"
#include "spgemv_cpu/spgemv_cpu_core.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMV {

// These templated structs are just little helpers to wrap the core functions.
// The operator() function is called in the dispatch_spgemv function to execute
// the correct function for the given types.
template <typename IT, typename VT> struct Init {
    int operator()(KernelContext context, Args *args, Flags *flags) {
        return SPGEMV_CPU::spgemv_initialize_cpu_core<IT, VT>(context, args,
                                                              flags);
    }
};

template <typename IT, typename VT> struct Apply {
    int operator()(KernelContext context, Args *args, Flags *flags) {
        return SPGEMV_CPU::spgemv_apply_cpu_core<IT, VT>(context, args, flags);
    }
};

template <typename IT, typename VT> struct Finalize {
    int operator()(KernelContext context, Args *args, Flags *flags) {
        return SPGEMV_CPU::spgemv_finalize_cpu_core<IT, VT>(context, args,
                                                            flags);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int spgemv_dispatch_cpu(KernelContext context, Args *args, Flags *flags) {
    switch (context.float_type) {
    case FLOAT32:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, float>()(context, args, flags);
        case UINT32:
            return Func<uint32_t, float>()(context, args, flags);
        case UINT64:
            return Func<uint64_t, float>()(context, args, flags);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FLOAT64:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, double>()(context, args, flags);
        case UINT32:
            return Func<uint32_t, double>()(context, args, flags);
        case UINT64:
            return Func<uint64_t, double>()(context, args, flags);
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
int spgemv_initialize_cpu(KernelContext context, Args *args, Flags *flags) {
    return spgemv_dispatch_cpu<Init>(context, args, flags);
}
int spgemv_apply_cpu(KernelContext context, Args *args, Flags *flags) {
    return spgemv_dispatch_cpu<Apply>(context, args, flags);
}
int spgemv_finalize_cpu(KernelContext context, Args *args, Flags *flags) {
    return spgemv_dispatch_cpu<Finalize>(context, args, flags);
}

} // namespace SPGEMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMV_CPU_HPP
