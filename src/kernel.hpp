#pragma once

#include "common.hpp"
#include "kernels/spgemm/spgemm_common.hpp"
#include "kernels/spmm/spmm_common.hpp"
#include "kernels/spmv/spmv_common.hpp"
#include "kernels/sptrsm/sptrsm_common.hpp"
#include "kernels/sptrsv/sptrsv_common.hpp"

#include <cstdarg>

namespace SMAX {

class Kernel {
    // TODO: Probably don't all need to be public
  public:
    std::unique_ptr<KERNELS::SPMV::Args> spmv_args;
    std::unique_ptr<KERNELS::SPMV::Flags> spmv_flags;
    std::unique_ptr<KERNELS::SPMM::Args> spmm_args;
    std::unique_ptr<KERNELS::SPMM::Flags> spmm_flags;
    std::unique_ptr<KERNELS::SPGEMM::Args> spgemm_args;
    std::unique_ptr<KERNELS::SPGEMM::Flags> spgemm_flags;
    std::unique_ptr<KERNELS::SPTRSV::Args> sptrsv_args;
    std::unique_ptr<KERNELS::SPTRSV::Flags> sptrsv_flags;
    std::unique_ptr<KERNELS::SPTRSM::Args> sptrsm_args;
    std::unique_ptr<KERNELS::SPTRSM::Flags> sptrsm_flags;

    std::unique_ptr<KernelContext> k_ctx;
    Timers *timers;

    Kernel(std::unique_ptr<KernelContext> _k_ctx) : k_ctx(std::move(_k_ctx)) {
        this->timers = new Timers;
#ifdef USE_TIMERS
        CREATE_STOPWATCH(initialize)
        CREATE_STOPWATCH(apply)
        CREATE_STOPWATCH(finalize)
#endif
    }
    virtual ~Kernel() { delete this->timers; };

    // Methods to override
    virtual int initialize(int A_offset, int B_offset, int C_offset) = 0;
    virtual int apply(int A_offset, int B_offset, int C_offset) = 0;
    virtual int finalize(int A_offset, int B_offset, int C_offset) = 0;

    int run(int A_offset = 0, int B_offset = 0, int C_offset = 0) {
        // DL 09.05.2025 NOTE: Since only a single memory address is
        // registered, the offsets are to access other parts of memory at
        // runtime (e.g. register A to kernel, but A[i*n_rows] needed)
        CHECK_ERROR(initialize(A_offset, B_offset, C_offset), "initialize");
        CHECK_ERROR(apply(A_offset, B_offset, C_offset), "apply");
        CHECK_ERROR(finalize(A_offset, B_offset, C_offset), "finalize");

        return 0;
    }

    // Flag setters to override
    virtual int set_mat_perm(bool) {
        std::cerr << "set_mat_perm not supported for this kernel.\n";
        return 1;
    }
    virtual int set_mat_upper_triang(bool) {
        std::cerr << "set_mat_upper_triang not supported for this kernel.\n";
        return 1;
    }
    virtual int set_mat_lower_triang(bool) {
        std::cerr << "set_mat_lower_triang not supported for this kernel.\n";
        return 1;
    }

    // Swapping utility to override
    virtual int swap_operands(void) {
        std::cerr << "swap_operands not supported for this kernel.\n";
        return 1;
    }

    // clang-format off
    // C-style variadic function arg list
    int register_A(...) {

        va_list user_args;
        va_start(user_args, this);

        switch (this->k_ctx->kernel_type) {
        case KernelType::SPMV: {
            int ret = KERNELS::SPMV::register_A(this->spmv_args->A, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPMM: {
            int ret = KERNELS::SPMM::register_A(this->spmm_args->A, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPGEMM: {
            int ret = KERNELS::SPGEMM::register_A(this->spgemm_args->A, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPTRSV: {
            int ret = KERNELS::SPTRSV::register_A(this->sptrsv_args->A, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPTRSM: {
            int ret = KERNELS::SPTRSM::register_A(this->sptrsm_args->A, user_args);
            va_end(user_args);
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        return 0;
    };

    // C-style variadic function arg list
    int register_B(...) {
        va_list user_args;
        va_start(user_args, this);

        switch (this->k_ctx->kernel_type) {
        case KernelType::SPMV: {
            int ret = KERNELS::SPMV::register_B(this->spmv_args->x, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPMM: {
            int ret = KERNELS::SPMM::register_B(this->spmm_args->X, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPGEMM: {
            int ret = KERNELS::SPGEMM::register_B(this->spgemm_args->B, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPTRSV: {
            int ret = KERNELS::SPTRSV::register_B(this->sptrsv_args->x, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPTRSM: {
            int ret = KERNELS::SPTRSM::register_B(this->sptrsm_args->X, user_args);
            va_end(user_args);
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        return 0;
    };

    // C-style variadic function arg list
    int register_C(...) {
        va_list user_args;
        va_start(user_args, this);

        switch (this->k_ctx->kernel_type) {
        case KernelType::SPMV: {
            int ret = KERNELS::SPMV::register_C(this->spmv_args->y, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPMM: {
            int ret = KERNELS::SPMM::register_C(this->spmm_args->Y, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPGEMM: {
            int ret = KERNELS::SPGEMM::register_C(this->spgemm_args->C, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPTRSV: {
            int ret = KERNELS::SPTRSV::register_C(this->sptrsv_args->y, user_args);
            va_end(user_args);
            return ret;
        }
        case KernelType::SPTRSM: {
            int ret = KERNELS::SPTRSM::register_C(this->sptrsm_args->Y, user_args);
            va_end(user_args);
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        return 0;
    };
    // clang-format on
};

} // namespace SMAX
