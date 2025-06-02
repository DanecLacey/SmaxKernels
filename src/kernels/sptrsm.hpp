#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "sptrsm/sptrsm_cpu.hpp"

namespace SMAX::KERNELS {

class SpTRSMKernel : public Kernel {
  public:
    std::unique_ptr<SPTRSM::Args> args;
    std::unique_ptr<SPTRSM::Flags> flags;

    using CpuFunc = int (*)(Timers *, KernelContext *, SPTRSM::Args *,
                            SPTRSM::Flags *);

    SpTRSMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    ~SpTRSMKernel() {}

    int _register_A(const std::vector<Variant> &args) override {
        if (args.size() != 6)
            throw std::runtime_error("SpTRSMKernel register_A expects 6 args");

        this->args->A->crs = std::make_unique<CRSMatrix>();

        this->args->A->crs->n_rows = std::get<int>(args[0]);
        this->args->A->crs->n_cols = std::get<int>(args[1]);
        this->args->A->crs->nnz = std::get<int>(args[2]);

        this->args->A->crs->col = std::get<void *>(args[3]);
        this->args->A->crs->row_ptr = std::get<void *>(args[4]);
        this->args->A->crs->val = std::get<void *>(args[5]);

        return 0;
    };

    int _register_B(const std::vector<Variant> &args) override {
        if (args.size() != 3)
            throw std::runtime_error("SpTRSMKernel register_B expects 3 args");

        this->args->X->n_rows = std::get<int>(args[0]);
        this->args->X->n_cols = std::get<int>(args[1]);
        this->args->X->val = std::get<void *>(args[2]);

        return 0;
    }

    int _register_C(const std::vector<Variant> &args) override {
        if (args.size() != 3)
            throw std::runtime_error("SpTRSMKernel register_C expects 3 args");

        this->args->Y->n_rows = std::get<int>(args[0]);
        this->args->Y->n_cols = std::get<int>(args[1]);
        this->args->Y->val = std::get<void *>(args[2]);

        return 0;
    }

    int set_vec_row_major(bool flag) override {
        this->flags->vec_row_major = flag;
        return 0;
    }

    int dispatch(CpuFunc cpu_func, const char *label) {
        IF_SMAX_DEBUG(if (!k_ctx || !args || !flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(timers, k_ctx.get(), args.get(), flags.get());
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
