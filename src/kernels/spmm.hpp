#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "spmm/spmm_cpu.hpp"

namespace SMAX::KERNELS {

class SpMMKernel : public Kernel {
  public:
    using CpuFunc = int (*)(Timers *, KernelContext *, SPMM::Args *,
                            SPMM::Flags *, int, int, int);

    SpMMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    int validate_A(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spmm_args->A->n_rows = *reinterpret_cast<int *>(args[0]);
        spmm_args->A->n_cols = *reinterpret_cast<int *>(args[1]);
        spmm_args->A->nnz = *reinterpret_cast<int *>(args[2]);

        spmm_args->A->col = reinterpret_cast<void **>(args[3]);
        spmm_args->A->row_ptr = reinterpret_cast<void **>(args[4]);
        spmm_args->A->val = reinterpret_cast<void **>(args[5]);

        return 0;
    }

    int validate_B(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spmm_args->X->n_rows = *reinterpret_cast<int *>(args[0]);
        spmm_args->X->n_cols = *reinterpret_cast<int *>(args[1]);
        spmm_args->X->val = reinterpret_cast<void **>(args[2]);

        return 0;
    }

    int validate_C(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spmm_args->Y->n_rows = *reinterpret_cast<int *>(args[0]);
        spmm_args->Y->n_cols = *reinterpret_cast<int *>(args[1]);
        spmm_args->Y->val = reinterpret_cast<void **>(args[2]);

        return 0;
    }

    int dispatch(CpuFunc cpu_func, const char *label, int A_offset,
                 int X_offset, int Y_offset) {
        IF_SMAX_DEBUG(if (!k_ctx || !spmm_args || !spmm_flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(timers, k_ctx.get(), spmm_args.get(),
                            spmm_flags.get(), A_offset, X_offset, Y_offset);
            break;
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int initialize(int A_offset, int X_offset, int Y_offset) override {
        return dispatch(SPMM::initialize_cpu, "spmm_initialize", A_offset,
                        X_offset, Y_offset);
    }

    int apply(int A_offset, int X_offset, int Y_offset) override {
        return dispatch(SPMM::apply_cpu, "spmm_apply", A_offset, X_offset,
                        Y_offset);
    }

    int finalize(int A_offset, int X_offset, int Y_offset) override {
        return dispatch(SPMM::finalize_cpu, "spmm_finalize", A_offset, X_offset,
                        Y_offset);
    }

    int swap_operands(void) override {
        std::swap(*spmm_args->X->val, *spmm_args->Y->val);
        return 0;
    }
};

} // namespace SMAX::KERNELS
