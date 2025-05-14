#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "sptrsm/sptrsm_cpu.hpp"

namespace SMAX::KERNELS {

class SpTRSMKernel : public Kernel {
  public:
    using CpuFunc = int (*)(KernelContext *, SPTRSM::Args *, SPTRSM::Flags *);

    SpTRSMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    int dispatch(CpuFunc cpu_func, const char *label) {
        IF_DEBUG(if (!k_ctx || !sptrsm_args || !sptrsm_flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(k_ctx.get(), sptrsm_args.get(), sptrsm_flags.get());
            break;
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int initialize(int A_offset, int X_offset, int Y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)X_offset;
        (void)Y_offset;
        return dispatch(SPTRSM::initialize_cpu, "sptrsm_initialize");
    }

    int apply(int A_offset, int X_offset, int Y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)X_offset;
        (void)Y_offset;
        return dispatch(SPTRSM::apply_cpu, "sptrsm_apply");
    }

    int finalize(int A_offset, int X_offset, int Y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)X_offset;
        (void)Y_offset;
        return dispatch(SPTRSM::finalize_cpu, "sptrsm_finalize");
    }
};

} // namespace SMAX::KERNELS
