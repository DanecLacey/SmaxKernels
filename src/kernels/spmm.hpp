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

    using Func = int (*)(Timers *, KernelContext *, SPMM::Args *, SPMM::Flags *,
                         ULL, ULL, ULL);

    SpMMKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    ~SpMMKernel() {}

    int _register_A(const std::vector<Variant> &args) override {
        if (args.size() != 6)
            throw std::runtime_error("SpMMKernel register_A expects 6 args");

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
        if (args.size() != 3)
            throw std::runtime_error("SpMMKernel register_B expects 3 args");

        this->args->X->n_rows = get_ull(args[0]);
        this->args->X->n_cols = get_ull(args[1]);
        this->args->X->val = std::get<void *>(args[2]);

        return 0;
    }

    int _register_C(const std::vector<Variant> &args) override {
        if (args.size() != 3)
            throw std::runtime_error("SpMMKernel register_C expects 3 args");

        this->args->Y->n_rows = get_ull(args[0]);
        this->args->Y->n_cols = get_ull(args[1]);
        this->args->Y->val = std::get<void *>(args[2]);

        return 0;
    }

    int set_vec_row_major(bool flag) override {
        this->flags->vec_row_major = flag;
        return 0;
    }

    int dispatch(Func func, const char *label, ULL A_offset, ULL X_offset,
                 ULL Y_offset) {
        IF_SMAX_DEBUG(if (!k_ctx || !args || !flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        switch (k_ctx->platform_type) {
        case PlatformType::CPU: {
            return func(timers, k_ctx.get(), args.get(), flags.get(), A_offset,
                        X_offset, Y_offset);
            break;
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int initialize(ULL A_offset, ULL X_offset, ULL Y_offset) override {
        return dispatch(SPMM::initialize_cpu, "spmm_initialize", A_offset,
                        X_offset, Y_offset);
    }

    int apply(ULL A_offset, ULL X_offset, ULL Y_offset) override {
        return dispatch(SPMM::apply_cpu, "spmm_apply", A_offset, X_offset,
                        Y_offset);
    }

    int finalize(ULL A_offset, ULL X_offset, ULL Y_offset) override {
        return dispatch(SPMM::finalize_cpu, "spmm_finalize", A_offset, X_offset,
                        Y_offset);
    }

    int swap_operands(void) override {
        std::swap(args->X, args->Y);
        return 0;
    }
};

} // namespace SMAX::KERNELS
