#ifndef SMAX_SPTRSV_CPU_HPP
#define SMAX_SPTRSV_CPU_HPP

#include "../../common.hpp"
#include "sptrsv_cpu/sptrsv_cpu_core.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPTRSV {

// These templated structs are just little helpers to wrap the core functions.
// The operator() function is called in the dispatch_sptrsv function to execute
// the correct function for the given types.
template <typename IT, typename VT> struct SpmvInit {
    int operator()(KernelContext context, SparseMatrix *A, DenseMatrix *x,
                   DenseMatrix *y) {
        return SPTRSV_CPU::sptrsv_initialize_cpu_core<IT, VT>(context, A, x, y);
    }
};

template <typename IT, typename VT> struct SpmvApply {
    int operator()(KernelContext context, SparseMatrix *A, DenseMatrix *x,
                   DenseMatrix *y) {
        return SPTRSV_CPU::sptrsv_apply_cpu_core<IT, VT>(context, A, x, y);
    }
};

template <typename IT, typename VT> struct SpmvFinalize {
    int operator()(KernelContext context, SparseMatrix *A, DenseMatrix *x,
                   DenseMatrix *y) {
        return SPTRSV_CPU::sptrsv_finalize_cpu_core<IT, VT>(context, A, x, y);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int sptrsv_dispatch_cpu(KernelContext context, SparseMatrix *A, DenseMatrix *x,
                        DenseMatrix *y) {
    switch (context.float_type) {
    case FLOAT32:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, float>()(context, A, x, y);
        case UINT32:
            return Func<uint32_t, float>()(context, A, x, y);
        case UINT64:
            return Func<uint64_t, float>()(context, A, x, y);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FLOAT64:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, double>()(context, A, x, y);
        case UINT32:
            return Func<uint32_t, double>()(context, A, x, y);
        case UINT64:
            return Func<uint64_t, double>()(context, A, x, y);
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
int sptrsv_initialize_cpu(KernelContext context, SparseMatrix *A,
                          DenseMatrix *x, DenseMatrix *y) {
    return sptrsv_dispatch_cpu<SpmvInit>(context, A, x, y);
}
int sptrsv_apply_cpu(KernelContext context, SparseMatrix *A, DenseMatrix *x,
                     DenseMatrix *y) {
    return sptrsv_dispatch_cpu<SpmvApply>(context, A, x, y);
}
int sptrsv_finalize_cpu(KernelContext context, SparseMatrix *A, DenseMatrix *x,
                        DenseMatrix *y) {
    return sptrsv_dispatch_cpu<SpmvFinalize>(context, A, x, y);
}

} // namespace SPTRSV
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSV_CPU_HPP
