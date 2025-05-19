#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "spmv/spmv_common.hpp"
#include "spmv/spmv_cpu.hpp"

namespace SMAX::KERNELS {

class SpMVKernel : public Kernel {
  public:
    std::unique_ptr<SPMV::Args> args;
    std::unique_ptr<SPMV::Flags> flags;

    using CpuFunc = int (*)(Timers *, KernelContext *, SPMV::Args *,
                            SPMV::Flags *, int, int, int);

    SpMVKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    ~SpMVKernel() {}

    int _register_A(const std::vector<Variant> &args) override {
        if (args.size() != 6)
            throw std::runtime_error("SpMVKernel register_A expects 6 args");

        this->args->A->n_rows = std::get<int>(args[0]);
        this->args->A->n_cols = std::get<int>(args[1]);
        this->args->A->nnz = std::get<int>(args[2]);

        this->args->A->col = std::get<void *>(args[3]);
        this->args->A->row_ptr = std::get<void *>(args[4]);
        this->args->A->val = std::get<void *>(args[5]);

        return 0;
    };

    int _register_B(const std::vector<Variant> &args) {
        if (args.size() != 2)
            throw std::runtime_error("SpMVKernel register_B expects 2 args");

        this->args->x->n_rows = std::get<int>(args[0]);
        this->args->x->val = std::get<void *>(args[1]);

        return 0;
    }

    int _register_C(const std::vector<Variant> &args) {
        if (args.size() != 2)
            throw std::runtime_error("SpMVKernel register_C expects 2 args");

        this->args->y->n_rows = std::get<int>(args[0]);
        this->args->y->val = std::get<void *>(args[1]);

        return 0;
    }

    // Dispatch kernel based on platform
    int dispatch(CpuFunc cpu_func, const char *label, int A_offset,
                 int x_offset, int y_offset) {
        IF_SMAX_DEBUG(if (!k_ctx || !args || !flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(timers, k_ctx.get(), args.get(), flags.get(),
                            A_offset, x_offset, y_offset);
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
        std::swap(args->x, args->y);
        return 0;
    }
};

} // namespace SMAX::KERNELS
