#ifndef SMAX_SPTRSM_CPU_HPP
#define SMAX_SPTRSM_CPU_HPP

#include "../../common.hpp"
#include "sptrsm_cpu/sptrsm_cpu_core.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSM {

// These templated structs are just little helpers to wrap the core functions.
// The operator() function is called in the dispatch_sptrsm function to execute
// the correct function for the given types.
template <typename IT, typename VT> struct Init {
    int operator()(KernelContext context, Args *args, Flags *flags) {
        return SPTRSM_CPU::sptrsm_initialize_cpu_core<IT, VT>(context, args,
                                                              flags);
    }
};

template <typename IT, typename VT> struct Apply {
    int operator()(KernelContext context, Args *args, Flags *flags) {
        return SPTRSM_CPU::sptrsm_apply_cpu_core<IT, VT>(context, args, flags);
    }
};

template <typename IT, typename VT> struct Finalize {
    int operator()(KernelContext context, Args *args, Flags *flags) {
        return SPTRSM_CPU::sptrsm_finalize_cpu_core<IT, VT>(context, args,
                                                            flags);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int sptrsm_dispatch_cpu(KernelContext context, Args *args, Flags *flags) {
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
int sptrsm_initialize_cpu(KernelContext context, Args *args, Flags *flags) {
    return sptrsm_dispatch_cpu<Init>(context, args, flags);
}
int sptrsm_apply_cpu(KernelContext context, Args *args, Flags *flags) {
    return sptrsm_dispatch_cpu<Apply>(context, args, flags);
}
int sptrsm_finalize_cpu(KernelContext context, Args *args, Flags *flags) {
    return sptrsm_dispatch_cpu<Finalize>(context, args, flags);
}

} // namespace SPTRSM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSM_CPU_HPP
