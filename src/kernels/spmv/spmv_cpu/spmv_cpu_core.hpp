#pragma once

#include "../../../common.hpp"
#include "spmv_cpu_crs_impl.hpp"

namespace SMAX::KERNELS::SPMV::SPMV_CPU {

template <typename IT, typename VT>
int initialize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                        Flags *flags, int A_offset, int x_offset,
                        int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_initialize_cpu_core"));
    IF_SMAX_TIME(timers->get("initialize")->start());

    // suppress unused warnings
    (void)timers;
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    // IF_SMAX_DEBUG_3(std::cout << "A->col: " << args->A->col << std::endl);
    // IF_SMAX_DEBUG_3(std::cout << "x->val: " << args->x->val << std::endl);
    // IF_SMAX_DEBUG_3(std::cout << "y->val: " << args->y->val << std::endl);

    IF_SMAX_TIME(timers->get("initialize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_initialize_cpu_core"));
    return 0;
};

template <typename IT, typename VT>
int apply_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                   Flags *flags, int A_offset, int x_offset, int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_apply_cpu_core"));
    IF_SMAX_TIME(timers->get("apply")->start());

    // suppress unused warnings
    (void)flags;
    (void)k_ctx;
    (void)A_offset;

    IF_SMAX_DEBUG_3(std::cout << "A_col pointer before deref: " << args->A->col
                              << std::endl);
    IF_SMAX_DEBUG_3(std::cout << "x pointer before deref: " << args->x
                              << std::endl);

    // Cast void pointers to the correct types with "as"
    // Dereference to get usable data
    int A_n_rows = args->A->n_rows;
    int A_n_cols = args->A->n_cols;
    IT *A_col = as<IT *>(args->A->col);
    IT *A_row_ptr = as<IT *>(args->A->row_ptr);
    VT *A_val = as<VT *>(args->A->val);
    VT *x = as<VT *>(args->x->val);
    VT *y = as<VT *>(args->y->val);

#if 1
    IF_SMAX_DEBUG_3(std::cout << "A_col pointer after deref: " << A_col
                              << std::endl);
    IF_SMAX_DEBUG_3(std::cout << "x pointer after deref: " << x << std::endl);
    naive_crs_spmv<IT, VT>(A_n_rows, A_n_cols, A_col, A_row_ptr, A_val,
                           x + x_offset, y + y_offset);
#endif

    IF_SMAX_TIME(timers->get("apply")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_apply_cpu_core"));
    return 0;
}

template <typename IT, typename VT>
int finalize_cpu_core(Timers *timers, KernelContext *k_ctx, Args *args,
                      Flags *flags, int A_offset, int x_offset, int y_offset) {
    IF_SMAX_DEBUG(ErrorHandler::log("Entering spmv_finalize_cpu_core"));
    IF_SMAX_TIME(timers->get("finalize")->start());

    // suppress unused warnings
    (void)timers;
    (void)k_ctx;
    (void)args;
    (void)flags;
    (void)A_offset;
    (void)x_offset;
    (void)y_offset;

    IF_SMAX_TIME(timers->get("finalize")->stop());
    IF_SMAX_DEBUG(ErrorHandler::log("Exiting spmv_finalize_cpu_core"));
    return 0;
}

} // namespace SMAX::KERNELS::SPMV::SPMV_CPU
