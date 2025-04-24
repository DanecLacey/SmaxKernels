#ifndef SPGEMM_CPU_CORE_HPP
#define SPGEMM_CPU_CORE_HPP

#include "../../../common.hpp"
#include "symbolic_phase.hpp"
#include "numerical_phase.hpp"

namespace SMAX
{

    template <typename IT, typename VT>
    int spgemm_initialize_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *A,
        SparseMatrix *B,
        SparseMatrix *C)
    {
        // TODO
        return 0;
    };

    template <typename IT, typename VT>
    int spgemm_apply_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *A,
        SparseMatrix *B,
        SparseMatrix *C)
    {
        symbolic_phase_cpu<IT, VT>(context, A, B, C);
        numerical_phase_cpu<IT, VT>(context, A, B, C);
        return 0;
    };

    template <typename IT, typename VT>
    int spgemm_finalize_cpu_core(
        SMAX::KernelContext context,
        SparseMatrix *A,
        SparseMatrix *B,
        SparseMatrix *C)
    {
        // TODO
        return 0;
    };

} // namespace SMAX

#endif // SPGEMM_CPU_CORE_HPP
