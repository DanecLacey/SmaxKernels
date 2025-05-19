#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "spmm/spmm_common.hpp"
#include "spmm/spmm_cpu.hpp"

namespace SMAX::KERNELS {

class SpMMKernel : public Kernel {
  public:
    std::unique_ptr<SPMM::Args> args;
    std::unique_ptr<SPMM::Flags> flags;

    using CpuFunc = int (*)(Timers *, KernelContext *, SPMM::Args *,
                            SPMM::Flags *, int, int, int);

    SpMMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    ~SpMMKernel() {}

    int _register_A(const std::vector<Variant> &args) override {
        if (args.size() != 6)
            throw std::runtime_error("SpMMKernel register_A expects 6 args");

        this->args->A->n_rows = std::get<int>(args[0]);
        this->args->A->n_cols = std::get<int>(args[1]);
        this->args->A->nnz = std::get<int>(args[2]);

        this->args->A->col = std::get<void *>(args[3]);
        this->args->A->row_ptr = std::get<void *>(args[4]);
        this->args->A->val = std::get<void *>(args[5]);

        return 0;
    };

    int _register_B(const std::vector<Variant> &args) {
        if (args.size() != 3)
            throw std::runtime_error("SpMMKernel register_B expects 3 args");

        this->args->X->n_rows = std::get<int>(args[0]);
        this->args->X->n_cols = std::get<int>(args[1]);
        this->args->X->val = std::get<void *>(args[2]);

        return 0;
    }

    int _register_C(const std::vector<Variant> &args) {
        if (args.size() != 3)
            throw std::runtime_error("SpMMKernel register_C expects 3 args");

        this->args->Y->n_rows = std::get<int>(args[0]);
        this->args->Y->n_cols = std::get<int>(args[1]);
        this->args->Y->val = std::get<void *>(args[2]);

        return 0;
    }

    int dispatch(CpuFunc cpu_func, const char *label, int A_offset,
                 int X_offset, int Y_offset) {
        IF_SMAX_DEBUG(if (!k_ctx || !args || !flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(timers, k_ctx.get(), args.get(), flags.get(),
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
        std::swap(args->X, args->Y);
        return 0;
    }
};

} // namespace SMAX::KERNELS
