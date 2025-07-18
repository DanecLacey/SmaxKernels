#pragma once

#include "../common.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "bspmv/bspmv_common.hpp"
#include "bspmv/bspmv_cpu.hpp"
#include "bspmv/bspmv_cuda.hpp"

namespace SMAX::KERNELS {

class BSpMVKernel : public Kernel {
  public:
    std::unique_ptr<BSPMV::Args> args;
    std::unique_ptr<BSPMV::Flags> flags;

    using Func = int (*)(Timers *, KernelContext *, BSPMV::Args *, BSPMV::Flags *,
                         ULL, ULL, ULL);

    BSpMVKernel(std::unique_ptr<KernelContext> k_ctx)
        : Kernel(std::move(k_ctx)) {}

    ~BSpMVKernel() {}

    int _register_A(const std::vector<Variant> &args) override {
        if (flags->is_mat_scs) {
          throw std::runtime_error(
            "Blocked Scs kernel not implemented yet");
//             if (args.size() != 13)
//                 throw std::runtime_error(
//                     "SpMVKernel register_A expects 13 Sell-C-sigma args");

//             this->args->A->scs = std::make_unique<SCSMatrix>();
// #if SMAX_CUDA_MODE
//             // Make device version of matrix
//             this->args->d_A->scs = std::make_unique<SCSMatrix>();
// #endif

//             this->args->A->scs->C = get_ull(args[0]);
//             this->args->A->scs->sigma = get_ull(args[1]);
//             this->args->A->scs->n_rows = get_ull(args[2]);
//             this->args->A->scs->n_rows_padded = get_ull(args[3]);
//             this->args->A->scs->n_cols = get_ull(args[4]);
//             this->args->A->scs->n_chunks = get_ull(args[5]);
//             this->args->A->scs->n_elements = get_ull(args[6]);
//             this->args->A->scs->nnz = get_ull(args[7]);

//             this->args->A->scs->chunk_ptr = std::get<void *>(args[8]);
//             this->args->A->scs->chunk_lengths = std::get<void *>(args[9]);
//             this->args->A->scs->col = std::get<void *>(args[10]);
//             this->args->A->scs->val = std::get<void *>(args[11]);
//             this->args->A->scs->perm = std::get<void *>(args[12]);

        } else {
            if (args.size() != 10)
                throw std::runtime_error(
                    "BSpMVKernel register_A expects 6 CRS args");

            this->args->A->crs = std::make_unique<BCRSMatrix>();
#if SMAX_CUDA_MODE
            // Make device version of matrix
            this->args->d_A->crs = std::make_unique<BCRSMatrix>();
#endif

            this->args->A->crs->n_rows = get_ull(args[0]);
            this->args->A->crs->n_cols = get_ull(args[1]);
            this->args->A->crs->nnz = get_ull(args[2]);
            this->args->A->crs->b_height = get_ull(args[3]);
            this->args->A->crs->b_width = get_ull(args[4]);
            this->args->A->crs->height_pad = get_ull(args[5]);
            this->args->A->crs->width_pad = get_ull(args[6]);

            this->args->A->crs->col = std::get<void *>(args[7]);
            this->args->A->crs->row_ptr = std::get<void *>(args[8]);
            this->args->A->crs->val = std::get<void *>(args[9]);
        }

        return 0;
    };

    int _register_B(const std::vector<Variant> &args) override {
        if (args.size() != 2)
            throw std::runtime_error("BSpMVKernel register_B expects 2 args");

//todo: we need to add padding here, do we?
        this->args->x->n_rows = get_ull(args[0]);
        this->args->x->val = std::get<void *>(args[1]);

        return 0;
    }

    int _register_C(const std::vector<Variant> &args) override {
        if (args.size() != 2)
            throw std::runtime_error("BSpMVKernel register_C expects 2 args");

        this->args->y->n_rows = get_ull(args[0]);
        this->args->y->val = std::get<void *>(args[1]);

        return 0;
    }

    int set_mat_scs(bool flag) override {
        this->flags->is_mat_scs = flag;
        return 0;
    }

    int set_block_column_major(bool flag) override {
        this->flags->is_block_column_major = flag;
        return 0;
    }

    int set_bspmv_kernel_implementation(BCRSKernelType impl) override {
        this->flags->kernel_type = impl;
        return 0;
    }

    // Dispatch kernel based on platform
    int dispatch(Func func, const char *label, ULL A_offset, ULL x_offset,
                 ULL y_offset) {
        IF_SMAX_DEBUG(if (!k_ctx || !args || !flags) {
            std::cerr << "Error: Null kernel state in " << label << "\n";
            return 1;
        });

        return func(timers, k_ctx.get(), args.get(), flags.get(), A_offset,
                    x_offset, y_offset);
    }

    int initialize(ULL A_offset, ULL x_offset, ULL y_offset) override {
        switch (this->k_ctx->platform_type) {
        case PlatformType::CPU: {
            return dispatch(BSPMV::initialize_cpu, "bspmv_finalize", A_offset,
                            x_offset, y_offset);
        }
        case PlatformType::CUDA: {
            return dispatch(BSPMV::initialize_cpu, "bspmv_finalize", A_offset,
                            x_offset, y_offset);
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int apply(ULL A_offset, ULL x_offset, ULL y_offset) override {
        switch (this->k_ctx->platform_type) {
        case PlatformType::CPU: {
            return dispatch(BSPMV::apply_cpu, "bspmv_apply", A_offset, x_offset,
                            y_offset);
        }
        case PlatformType::CUDA: {
            return dispatch(BSPMV::apply_cpu, "bspmv_apply", A_offset, x_offset,
                            y_offset);
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int finalize(ULL A_offset, ULL x_offset, ULL y_offset) override {
        switch (this->k_ctx->platform_type) {
        case PlatformType::CPU: {
            return dispatch(BSPMV::finalize_cpu, "bspmv_finalize", A_offset,
                            x_offset, y_offset);
        }
        case PlatformType::CUDA: {
            return dispatch(BSPMV::finalize_cpu, "bspmv_finalize", A_offset,
                            x_offset, y_offset);
        }
        default:
            std::cerr << "Error: Platform not supported\n";
            return 1;
        }
    }

    int swap_operands(void) override {
        std::swap(args->x, args->y);
        return 0;
    }
};

} // namespace SMAX::KERNELS