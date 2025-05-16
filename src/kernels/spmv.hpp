#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "spmv/spmv_cpu.hpp"

namespace SMAX::KERNELS {

class SpMVKernel : public Kernel {
  public:
    using CpuFunc = int (*)(Timers *, KernelContext *, SPMV::Args *,
                            SPMV::Flags *, int, int, int);

    SpMVKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    ~SpMVKernel() {}

    // Dispatch kernel based on platform
    int dispatch(CpuFunc cpu_func, const char *label, int A_offset,
                 int x_offset, int y_offset) {
        IF_SMAX_DEBUG(if (!k_ctx || !spmv_args || !spmv_flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(timers, k_ctx.get(), spmv_args.get(),
                            spmv_flags.get(), A_offset, x_offset, y_offset);
            break;
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int initialize(int A_offset, int x_offset, int y_offset) override {
        return dispatch(SPMV::initialize_cpu, "spmv_initialize", A_offset,
                        x_offset, y_offset);
    }

    int apply(int A_offset, int x_offset, int y_offset) override {
        return dispatch(SPMV::apply_cpu, "spmv_apply", A_offset, x_offset,
                        y_offset);
    }

    int finalize(int A_offset, int x_offset, int y_offset) override {
        return dispatch(SPMV::finalize_cpu, "spmv_finalize", A_offset, x_offset,
                        y_offset);
    }

    int swap_operands(void) override {
        std::swap(*spmv_args->x->val, *spmv_args->y->val);
        return 0;
    }
};

} // namespace SMAX::KERNELS
