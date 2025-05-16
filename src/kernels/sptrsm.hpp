#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "sptrsm/sptrsm_cpu.hpp"

namespace SMAX::KERNELS {

class SpTRSMKernel : public Kernel {
  public:
    using CpuFunc = int (*)(Timers *, KernelContext *, SPTRSM::Args *,
                            SPTRSM::Flags *);

    SpTRSMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    int validate_A(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        sptrsm_args->A->n_rows = *reinterpret_cast<int *>(args[0]);
        sptrsm_args->A->n_cols = *reinterpret_cast<int *>(args[1]);
        sptrsm_args->A->nnz = *reinterpret_cast<int *>(args[2]);

        sptrsm_args->A->col = reinterpret_cast<void **>(args[3]);
        sptrsm_args->A->row_ptr = reinterpret_cast<void **>(args[4]);
        sptrsm_args->A->val = reinterpret_cast<void **>(args[5]);

        return 0;
    }

    int validate_B(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        sptrsm_args->X->n_rows = *reinterpret_cast<int *>(args[0]);
        sptrsm_args->X->n_cols = *reinterpret_cast<int *>(args[1]);
        sptrsm_args->X->val = reinterpret_cast<void **>(args[2]);

        return 0;
    }

    int validate_C(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        sptrsm_args->Y->n_rows = *reinterpret_cast<int *>(args[0]);
        sptrsm_args->Y->n_cols = *reinterpret_cast<int *>(args[1]);
        sptrsm_args->Y->val = reinterpret_cast<void **>(args[2]);

        return 0;
    }

    int dispatch(CpuFunc cpu_func, const char *label) {
        IF_SMAX_DEBUG(if (!k_ctx || !sptrsm_args || !sptrsm_flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(timers, k_ctx.get(), sptrsm_args.get(),
                            sptrsm_flags.get());
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
