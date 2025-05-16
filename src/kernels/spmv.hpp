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

    int validate_A(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spmv_args->A->n_rows = *reinterpret_cast<int *>(args[0]);
        spmv_args->A->n_cols = *reinterpret_cast<int *>(args[1]);
        spmv_args->A->nnz = *reinterpret_cast<int *>(args[2]);

        spmv_args->A->col = reinterpret_cast<void **>(args[3]);
        spmv_args->A->row_ptr = reinterpret_cast<void **>(args[4]);
        spmv_args->A->val = reinterpret_cast<void **>(args[5]);

        return 0;
    }

    int validate_B(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spmv_args->x->n_rows = *reinterpret_cast<int *>(args[0]);
        spmv_args->x->val = reinterpret_cast<void **>(args[1]);

        return 0;
    }

    int validate_C(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spmv_args->y->n_rows = *reinterpret_cast<int *>(args[0]);
        spmv_args->y->val = reinterpret_cast<void **>(args[1]);

        return 0;
    }

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
