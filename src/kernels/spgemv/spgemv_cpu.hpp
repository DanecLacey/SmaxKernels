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
template <typename IT, typename VT> struct SPGEMVInit {
    int operator()(KernelContext context, SparseMatrix *A, SparseVector *spX,
                   SparseVectorRef *spY_ref) {
        return SPGEMV_CPU::spgemv_initialize_cpu_core<IT, VT>(context, A, spX,
                                                              spY_ref);
    }
};

template <typename IT, typename VT> struct SPGEMVApply {
    int operator()(KernelContext context, SparseMatrix *A, SparseVector *spX,
                   SparseVectorRef *spY_ref) {
        return SPGEMV_CPU::spgemv_apply_cpu_core<IT, VT>(context, A, spX,
                                                         spY_ref);
    }
};

template <typename IT, typename VT> struct SPGEMVFinalize {
    int operator()(KernelContext context, SparseMatrix *A, SparseVector *spX,
                   SparseVectorRef *spY_ref) {
        return SPGEMV_CPU::spgemv_finalize_cpu_core<IT, VT>(context, A, spX,
                                                            spY_ref);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int spgemv_dispatch_cpu(KernelContext context, SparseMatrix *A,
                        SparseVector *spX, SparseVectorRef *spY_ref) {
    switch (context.float_type) {
    case FLOAT32:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, float>()(context, A, spX, spY_ref);
        case UINT32:
            return Func<uint32_t, float>()(context, A, spX, spY_ref);
        case UINT64:
            return Func<uint64_t, float>()(context, A, spX, spY_ref);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FLOAT64:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, double>()(context, A, spX, spY_ref);
        case UINT32:
            return Func<uint32_t, double>()(context, A, spX, spY_ref);
        case UINT64:
            return Func<uint64_t, double>()(context, A, spX, spY_ref);
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
int spgemv_initialize_cpu(KernelContext context, SparseMatrix *A,
                          SparseVector *spX, SparseVectorRef *spY_ref) {
    return spgemv_dispatch_cpu<SPGEMVInit>(context, A, spX, spY_ref);
}
int spgemv_apply_cpu(KernelContext context, SparseMatrix *A, SparseVector *spX,
                     SparseVectorRef *spY_ref) {
    return spgemv_dispatch_cpu<SPGEMVApply>(context, A, spX, spY_ref);
}
int spgemv_finalize_cpu(KernelContext context, SparseMatrix *A,
                        SparseVector *spX, SparseVectorRef *spY_ref) {
    return spgemv_dispatch_cpu<SPGEMVFinalize>(context, A, spX, spY_ref);
}

} // namespace SPGEMV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPGEMV_CPU_HPP
