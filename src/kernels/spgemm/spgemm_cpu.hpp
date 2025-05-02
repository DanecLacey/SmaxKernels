#ifndef SPGEMM_CPU_HPP
#define SPGEMM_CPU_HPP

#include "../../common.hpp"
#include "spgemm_cpu/spgemm_cpu_core.hpp"

namespace SMAX {
namespace KERNELS {
namespace SPGEMM {

// These templated structs are just little helpers to wrap the core functions.
// The operator() function is called in the dispatch_spgemm function to execute
// the correct function for the given types.
template <typename IT, typename VT> struct SPGEMMInit {
    int operator()(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                   SparseMatrixRef *C_ref) {
        return SPGEMM_CPU::spgemm_initialize_cpu_core<IT, VT>(context, A, B,
                                                              C_ref);
    }
};

template <typename IT, typename VT> struct SPGEMMApply {
    int operator()(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                   SparseMatrixRef *C_ref) {
        return SPGEMM_CPU::spgemm_apply_cpu_core<IT, VT>(context, A, B, C_ref);
    }
};

template <typename IT, typename VT> struct SPGEMMFinalize {
    int operator()(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                   SparseMatrixRef *C_ref) {
        return SPGEMM_CPU::spgemm_finalize_cpu_core<IT, VT>(context, A, B,
                                                            C_ref);
    }
};

// The dispatcher function uses the above () operator with the correct
// integer and floating point types.
template <template <typename IT, typename VT> class Func>
int spgemm_dispatch_cpu(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                        SparseMatrixRef *C_ref) {
    switch (context.float_type) {
    case FLOAT32:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, float>()(context, A, B, C_ref);
        case UINT32:
            return Func<uint32_t, float>()(context, A, B, C_ref);
        case UINT64:
            return Func<uint64_t, float>()(context, A, B, C_ref);
        default:
            std::cerr << "Error: Int type not supported\n";
            return 1;
        }
    case FLOAT64:
        switch (context.int_type) {
        case UINT16:
            return Func<uint16_t, double>()(context, A, B, C_ref);
        case UINT32:
            return Func<uint32_t, double>()(context, A, B, C_ref);
        case UINT64:
            return Func<uint64_t, double>()(context, A, B, C_ref);
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
int spgemm_initialize_cpu(KernelContext context, SparseMatrix *A,
                          SparseMatrix *B, SparseMatrixRef *C_ref) {
    return spgemm_dispatch_cpu<SPGEMMInit>(context, A, B, C_ref);
}
int spgemm_apply_cpu(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                     SparseMatrixRef *C_ref) {
    return spgemm_dispatch_cpu<SPGEMMApply>(context, A, B, C_ref);
}
int spgemm_finalize_cpu(KernelContext context, SparseMatrix *A, SparseMatrix *B,
                        SparseMatrixRef *C_ref) {
    return spgemm_dispatch_cpu<SPGEMMFinalize>(context, A, B, C_ref);
}

} // namespace SPGEMM
} // namespace KERNELS
} // namespace SMAX

#endif // SPGEMM_CPU_HPP
