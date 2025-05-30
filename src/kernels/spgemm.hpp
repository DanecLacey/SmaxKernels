#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "spgemm/spgemm_cpu.hpp"

namespace SMAX::KERNELS {

class SpGEMMKernel : public Kernel {
  public:
    std::unique_ptr<SPGEMM::Args> args;
    std::unique_ptr<SPGEMM::Flags> flags;

    using CpuFunc = int (*)(Timers *, KernelContext *, SPGEMM::Args *,
                            SPGEMM::Flags *);

    SpGEMMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {
        CREATE_SMAX_STOPWATCH(symbolic_phase)
        CREATE_SMAX_STOPWATCH(Symbolic_Setup)
        CREATE_SMAX_STOPWATCH(Symbolic_Gustavson)
        CREATE_SMAX_STOPWATCH(Alloc_C)
        CREATE_SMAX_STOPWATCH(Compress)
        CREATE_SMAX_STOPWATCH(numerical_phase)
        CREATE_SMAX_STOPWATCH(Numerical_Setup)
        CREATE_SMAX_STOPWATCH(Numerical_Gustavson)
    }

    ~SpGEMMKernel() {}

    int _register_A(const std::vector<Variant> &args) override {
        if (args.size() != 6)
            throw std::runtime_error("SpGEMMKernel register_A expects 6 args");

        this->args->A->crs = std::make_unique<CRSMatrix>();

        this->args->A->crs->n_rows = std::get<int>(args[0]);
        this->args->A->crs->n_cols = std::get<int>(args[1]);
        this->args->A->crs->nnz = std::get<int>(args[2]);

        this->args->A->crs->col = std::get<void *>(args[3]);
        this->args->A->crs->row_ptr = std::get<void *>(args[4]);
        this->args->A->crs->val = std::get<void *>(args[5]);

        return 0;
    };

    int _register_B(const std::vector<Variant> &args) {
        if (args.size() != 6)
            throw std::runtime_error("SpGEMMKernel register_B expects 6 args");

        this->args->B->crs = std::make_unique<CRSMatrix>();

        this->args->B->crs->n_rows = std::get<int>(args[0]);
        this->args->B->crs->n_cols = std::get<int>(args[1]);
        this->args->B->crs->nnz = std::get<int>(args[2]);

        this->args->B->crs->col = std::get<void *>(args[3]);
        this->args->B->crs->row_ptr = std::get<void *>(args[4]);
        this->args->B->crs->val = std::get<void *>(args[5]);

        return 0;
    }

    int _register_C(const std::vector<Variant> &args) {
        if (args.size() != 6)
            throw std::runtime_error("SpGEMMKernel register_C expects 6 args");

        this->args->C->crs = std::make_unique<CRSMatrixRef>();

        this->args->C->crs->n_rows = std::get<void *>(args[0]);
        this->args->C->crs->n_cols = std::get<void *>(args[1]);
        this->args->C->crs->nnz = std::get<void *>(args[2]);

        this->args->C->crs->col = std::get<void **>(args[3]);
        this->args->C->crs->row_ptr = std::get<void **>(args[4]);
        this->args->C->crs->val = std::get<void **>(args[5]);

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

    int initialize(int A_offset, int B_offset, int C_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)B_offset;
        (void)C_offset;
        return dispatch(SPGEMM::initialize_cpu, "spgemm_initialize");
    }

    int apply(int A_offset, int B_offset, int C_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)B_offset;
        (void)C_offset;
        return dispatch(SPGEMM::apply_cpu, "spgemm_apply");
    }

    int finalize(int A_offset, int B_offset, int C_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)B_offset;
        (void)C_offset;
        return dispatch(SPGEMM::finalize_cpu, "spgemm_finalize");
    }
};

} // namespace SMAX::KERNELS
