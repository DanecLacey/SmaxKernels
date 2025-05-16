#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "sptrsv/sptrsv_cpu.hpp"

namespace SMAX::KERNELS {

class SpTRSVKernel : public Kernel {
  public:
    using CpuFunc = int (*)(Timers *, KernelContext *, SPTRSV::Args *,
                            SPTRSV::Flags *);

    SpTRSVKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    int validate_A(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        sptrsv_args->A->n_rows = *reinterpret_cast<int *>(args[0]);
        sptrsv_args->A->n_cols = *reinterpret_cast<int *>(args[1]);
        sptrsv_args->A->nnz = *reinterpret_cast<int *>(args[2]);

        sptrsv_args->A->col = reinterpret_cast<void **>(args[3]);
        sptrsv_args->A->row_ptr = reinterpret_cast<void **>(args[4]);
        sptrsv_args->A->val = reinterpret_cast<void **>(args[5]);

        return 0;
    }

    int validate_B(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        sptrsv_args->x->n_rows = *reinterpret_cast<int *>(args[0]);
        sptrsv_args->x->val = reinterpret_cast<void **>(args[1]);

        return 0;
    }

    int validate_C(const std::vector<void *> &args) override {
        if (args.size() != 6) {
            // TODO: throw error
            // std::cerr << "SPMV::register_A requires exactly 6 arguments\n";
            // return 1;
        }

        // Manually cast from void* to expected types at runtime
        sptrsv_args->y->n_rows = *reinterpret_cast<int *>(args[0]);
        sptrsv_args->y->val = reinterpret_cast<void **>(args[1]);

        return 0;
    }

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
        IF_SMAX_DEBUG(if (!k_ctx || !sptrsv_args || !sptrsv_flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return cpu_func(timers, k_ctx.get(), sptrsv_args.get(),
                            sptrsv_flags.get());
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
