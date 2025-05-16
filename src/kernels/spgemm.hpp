#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "spgemm/spgemm_cpu.hpp"

namespace SMAX::KERNELS {

class SpGEMMKernel : public Kernel {
  public:
    using CpuFunc = int (*)(Timers *, KernelContext *, SPGEMM::Args *,
                            SPGEMM::Flags *);

    SpGEMMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {
        CREATE_SMAX_STOPWATCH(symbolic_phase)
        CREATE_SMAX_STOPWATCH(numerical_phase)
    }

    int validate_A(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spgemm_args->A->n_rows = *reinterpret_cast<int *>(args[0]);
        spgemm_args->A->n_cols = *reinterpret_cast<int *>(args[1]);
        spgemm_args->A->nnz = *reinterpret_cast<int *>(args[2]);

        spgemm_args->A->col = reinterpret_cast<void **>(args[3]);
        spgemm_args->A->row_ptr = reinterpret_cast<void **>(args[4]);
        spgemm_args->A->val = reinterpret_cast<void **>(args[5]);

        return 0;
    }

    int validate_B(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spgemm_args->B->n_rows = *reinterpret_cast<int *>(args[0]);
        spgemm_args->B->n_cols = *reinterpret_cast<int *>(args[1]);
        spgemm_args->B->nnz = *reinterpret_cast<int *>(args[2]);

        spgemm_args->B->col = reinterpret_cast<void **>(args[3]);
        spgemm_args->B->row_ptr = reinterpret_cast<void **>(args[4]);
        spgemm_args->B->val = reinterpret_cast<void **>(args[5]);

        return 0;
    }

    int validate_C(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        spgemm_args->C->n_rows = reinterpret_cast<int *>(args[0]);
        spgemm_args->C->n_cols = reinterpret_cast<int *>(args[1]);
        spgemm_args->C->nnz = reinterpret_cast<int *>(args[2]);

        spgemm_args->C->col = reinterpret_cast<void **>(args[3]);
        spgemm_args->C->row_ptr = reinterpret_cast<void **>(args[4]);
        spgemm_args->C->val = reinterpret_cast<void **>(args[5]);

        return 0;
    }

    int dispatch(CpuFunc cpu_func, const char *label) {
        IF_SMAX_DEBUG(if (!k_ctx || !spgemm_args || !spgemm_flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(timers, k_ctx.get(), spgemm_args.get(),
                            spgemm_flags.get());
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
