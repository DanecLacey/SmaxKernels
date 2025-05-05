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
template <typename IT, typename VT> struct SpmvInit {
    int operator()(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                   DenseMatrix *Y) {
        return SPTRSM_CPU::sptrsm_initialize_cpu_core<IT, VT>(context, A, X, Y);
    }
};

template <typename IT, typename VT> struct SpmvApply {
    int operator()(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                   DenseMatrix *Y) {
        return SPTRSM_CPU::sptrsm_apply_cpu_core<IT, VT>(context, A, X, Y);
    }
};

template <typename IT, typename VT> struct SpmvFinalize {
    int operator()(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                   DenseMatrix *Y) {
        return SPTRSM_CPU::sptrsm_finalize_cpu_core<IT, VT>(context, A, X, Y);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int sptrsm_dispatch_cpu(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                        DenseMatrix *Y) {
    switch (context.float_type) {
    case FLOAT32:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, float>()(context, A, X, Y);
        case UINT32:
            return Func<uint32_t, float>()(context, A, X, Y);
        case UINT64:
            return Func<uint64_t, float>()(context, A, X, Y);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FLOAT64:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, double>()(context, A, X, Y);
        case UINT32:
            return Func<uint32_t, double>()(context, A, X, Y);
        case UINT64:
            return Func<uint64_t, double>()(context, A, X, Y);
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
int sptrsm_initialize_cpu(KernelContext context, SparseMatrix *A,
                          DenseMatrix *X, DenseMatrix *Y) {
    return sptrsm_dispatch_cpu<SpmvInit>(context, A, X, Y);
}
int sptrsm_apply_cpu(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                     DenseMatrix *Y) {
    return sptrsm_dispatch_cpu<SpmvApply>(context, A, X, Y);
}
int sptrsm_finalize_cpu(KernelContext context, SparseMatrix *A, DenseMatrix *X,
                        DenseMatrix *Y) {
    return sptrsm_dispatch_cpu<SpmvFinalize>(context, A, X, Y);
}

} // namespace SPTRSM
} // namespace KERNELS
} // namespace SMAX

#endif // SMAX_SPTRSM_CPU_HPP
