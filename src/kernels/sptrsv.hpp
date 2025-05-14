#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "sptrsv/sptrsv_cpu.hpp"

namespace SMAX::KERNELS {

class SpTRSVKernel : public Kernel {
  public:
    using CpuFunc = int (*)(KernelContext *, SPTRSV::Args *, SPTRSV::Flags *);

    SpTRSVKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    // Flag setters
    int set_mat_perm(bool flag) override {
        this->sptrsv_flags->mat_permuted = flag;
        return 0;
    }
    int set_mat_upper_triang(bool flag) override {
        this->sptrsv_flags->mat_upper_triang = flag;
        return 0;
    }
    int set_mat_lower_triang(bool flag) override {
        this->sptrsv_flags->mat_lower_triang = flag;
        return 0;
    }

    int dispatch(CpuFunc cpu_func, const char *label) {
        IF_DEBUG(if (!k_ctx || !sptrsv_args || !sptrsv_flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(k_ctx.get(), sptrsv_args.get(), sptrsv_flags.get());
            break;
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int initialize(int A_offset, int x_offset, int y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)x_offset;
        (void)y_offset;
        return dispatch(SPTRSV::initialize_cpu, "sptrsv_initialize");
    }

    int apply(int A_offset, int x_offset, int y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)x_offset;
        (void)y_offset;
        return dispatch(SPTRSV::apply_cpu, "sptrsv_apply");
    }

    int finalize(int A_offset, int x_offset, int y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)x_offset;
        (void)y_offset;
        return dispatch(SPTRSV::finalize_cpu, "sptrsv_finalize");
    }
};

} // namespace SMAX::KERNELS
