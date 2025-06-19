#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "sptrsv/sptrsv_cpu.hpp"

namespace SMAX::KERNELS {

class SpTRSVKernel : public Kernel {
  public:
    std::unique_ptr<SPTRSV::Args> args;
    std::unique_ptr<SPTRSV::Flags> flags;

    using Func = int (*)(Timers *, KernelContext *, SPTRSV::Args *,
                         SPTRSV::Flags *);

    SpTRSVKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    ~SpTRSVKernel() {}

    int _register_A(const std::vector<Variant> &args) override {
        if (args.size() != 6)
            throw std::runtime_error("SpTRSVKernel register_A expects 6 args");

        this->args->A->crs = std::make_unique<CRSMatrix>();

        this->args->A->crs->n_rows = get_ull(args[0]);
        this->args->A->crs->n_cols = get_ull(args[1]);
        this->args->A->crs->nnz = get_ull(args[2]);

        this->args->A->crs->col = std::get<void *>(args[3]);
        this->args->A->crs->row_ptr = std::get<void *>(args[4]);
        this->args->A->crs->val = std::get<void *>(args[5]);

        return 0;
    };

    int _register_B(const std::vector<Variant> &args) override {
        if (args.size() != 2)
            throw std::runtime_error("SpTRSVKernel register_B expects 2 args");

        this->args->x->n_rows = get_ull(args[0]);
        this->args->x->val = std::get<void *>(args[1]);

        return 0;
    }

    int _register_C(const std::vector<Variant> &args) override {
        if (args.size() != 2)
            throw std::runtime_error("SpTRSVKernel register_C expects 2 args");

        this->args->y->n_rows = get_ull(args[0]);
        this->args->y->val = std::get<void *>(args[1]);

        return 0;
    }

    // Flag setters
    int set_mat_perm(bool flag) override {
        this->flags->mat_permuted = flag;
        return 0;
    }
    int set_mat_upper_triang(bool flag) override {
        this->flags->mat_upper_triang = flag;
        return 0;
    }
    int set_mat_lower_triang(bool flag) override {
        this->flags->mat_lower_triang = flag;
        return 0;
    }

    int dispatch(Func func, const char *label) {
        IF_SMAX_DEBUG(if (!k_ctx || !args || !flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return func(timers, k_ctx.get(), args.get(), flags.get());
            break;
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int initialize(ULL A_offset, ULL x_offset, ULL y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)x_offset;
        (void)y_offset;
        return dispatch(SPTRSV::initialize_cpu, "sptrsv_initialize");
    }

    int apply(ULL A_offset, ULL x_offset, ULL y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)x_offset;
        (void)y_offset;
        return dispatch(SPTRSV::apply_cpu, "sptrsv_apply");
    }

    int finalize(ULL A_offset, ULL x_offset, ULL y_offset) override {
        // suppress unused warnings
        (void)A_offset;
        (void)x_offset;
        (void)y_offset;
        return dispatch(SPTRSV::finalize_cpu, "sptrsv_finalize");
    }
};

} // namespace SMAX::KERNELS
