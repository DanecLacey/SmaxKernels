#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "spmm/spmm_cpu.hpp"

namespace SMAX::KERNELS {

class SpMMKernel : public Kernel {
  public:
    using CpuFunc = int (*)(KernelContext *, SPMM::Args *, SPMM::Flags *, int,
                            int, int);

    SpMMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    int dispatch(CpuFunc cpu_func, const char *label, int A_offset,
                 int X_offset, int Y_offset) {
        IF_DEBUG(if (!k_ctx || !spmm_args || !spmm_flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(k_ctx.get(), spmm_args.get(), spmm_flags.get(),
                            A_offset, X_offset, Y_offset);
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
