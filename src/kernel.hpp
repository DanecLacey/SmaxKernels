#ifndef SMAX_KERNEL_HPP
#define SMAX_KERNEL_HPP

#include "common.hpp"
#include "kernels/spgemm.hpp"
#include "kernels/spgemv.hpp"
#include "kernels/spmm.hpp"
#include "kernels/spmv.hpp"
#include "kernels/sptrsm.hpp"
#include "kernels/sptrsv.hpp"

#include <cstdarg>

namespace SMAX {

class Kernel {
  private:
    KernelType kernel_type;
    PlatformType platform;
    IntType int_type;
    FloatType float_type;

  public:
    // TODO: Do these need to be public?
    SparseMatrix *A;
    SparseMatrix *B;
    SparseMatrix *C;
    SparseMatrixRef *C_ref;
    DenseMatrix *dX;
    DenseMatrix *dY;
    SparseVector *spX;
    SparseVector *spY;
    SparseVectorRef *spY_ref;
    KernelContext context;

    Kernel(KernelType kernel_type, PlatformType platform, IntType int_type,
           FloatType float_type) {
        this->context =
            KernelContext{kernel_type, platform, int_type, float_type};
    }

    ~Kernel() {
        delete A;
        delete B;
        delete C;
        delete C_ref;
        delete dX;
        delete dY;
        delete spX;
        delete spY;
        delete spY_ref;
    }

    // Register A, B, C matrices and x, y vectors
    // These functions are variadic and take a variable number of arguments
    int register_A(...) {
        // this->timers->register_A_time->start();

        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type) {
        case SPMV: {
            int ret = SMAX::KERNELS::spmv_register_A(this->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPMM: {
            int ret = SMAX::KERNELS::spmm_register_A(this->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPGEMV: {
            int ret = SMAX::KERNELS::spgemv_register_A(this->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPGEMM: {
            int ret = SMAX::KERNELS::spgemm_register_A(this->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPTRSV: {
            int ret = SMAX::KERNELS::sptrsv_register_A(this->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPTRSM: {
            int ret = SMAX::KERNELS::sptrsm_register_A(this->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        // this->timers->register_A_time->stop();
        return 0;
    }

    int register_B(...) {
        // this->timers->register_B_time->start();
        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type) {
        case SPMV: {
            int ret = SMAX::KERNELS::spmv_register_B(this->dX, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPMM: {
            int ret = SMAX::KERNELS::spmm_register_B(this->dX, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPGEMV: {
            int ret = SMAX::KERNELS::spgemv_register_B(this->spX, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPGEMM: {
            int ret = SMAX::KERNELS::spgemm_register_B(this->B, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPTRSV: {
            int ret = SMAX::KERNELS::sptrsv_register_B(this->dX, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPTRSM: {
            int ret = SMAX::KERNELS::sptrsm_register_B(this->dX, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        // this->timers->register_B_time->stop();
        return 0;
    }

    int register_C(...) {
        // this->timers->register_C_time->start();
        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type) {
        case SPMV: {
            int ret = SMAX::KERNELS::spmv_register_C(this->dY, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPMM: {
            int ret = SMAX::KERNELS::spmm_register_C(this->dY, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPGEMV: {
            int ret =
                SMAX::KERNELS::spgemv_register_C(this->spY_ref, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPGEMM: {
            int ret = SMAX::KERNELS::spgemm_register_C(this->C_ref, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPTRSV: {
            int ret = SMAX::KERNELS::sptrsv_register_C(this->dY, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPTRSM: {
            int ret = SMAX::KERNELS::sptrsm_register_C(this->dY, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        // this->timers->register_C_time->stop();
        return 0;
    }

    int dispatch(
        std::function<int(KernelContext)> func_spmv, const char *label_spmv,
        std::function<int(KernelContext)> func_spmm, const char *label_spmm,
        std::function<int(KernelContext)> func_spgemv, const char *label_spgemv,
        std::function<int(KernelContext)> func_spgemm, const char *label_spgemm,
        std::function<int(KernelContext)> func_sptrsv, const char *label_sptrsv,
        std::function<int(KernelContext)> func_sptrsm,
        const char *label_sptrsm) {
        switch (this->context.kernel_type) {
        case SMAX::SPMV:
            CHECK_ERROR(func_spmv(this->context), label_spmv);
            break;
        case SMAX::SPMM:
            CHECK_ERROR(func_spmm(this->context), label_spmm);
            break;
        case SMAX::SPGEMV:
            CHECK_ERROR(func_spgemv(this->context), label_spgemv);
            break;
        case SMAX::SPGEMM:
            CHECK_ERROR(func_spgemm(this->context), label_spgemm);
            break;
        case SMAX::SPTRSV:
            CHECK_ERROR(func_sptrsv(this->context), label_sptrsv);
            break;
        case SMAX::SPTRSM:
            CHECK_ERROR(func_sptrsm(this->context), label_sptrsm);
            break;
        default:
            std::cerr << "Error: Kernel: " << this->kernel_type
                      << " not supported\n";
            return 1;
        }
        return 0;
    }

    int initialize(int A_offset = 0, int B_offset = 0, int C_offset = 0) {
        // this->timers->initialize_time->start();
        int ret = dispatch(
            [this, A_offset, B_offset, C_offset](KernelContext context) {
                return SMAX::KERNELS::spmv_initialize(
                    context, this->A, this->dX, this->dY, A_offset, B_offset,
                    C_offset);
            },
            "spmv_initialize",
            [this, A_offset, B_offset, C_offset](KernelContext context) {
                return SMAX::KERNELS::spmm_initialize(
                    context, this->A, this->dX, this->dY, A_offset, B_offset,
                    C_offset);
            },
            "spmm_initialize",
            [this](KernelContext context) {
                return SMAX::KERNELS::spgemv_initialize(
                    context, this->A, this->spX, this->spY_ref);
            },
            "spgemv_initialize",
            [this](KernelContext context) {
                return SMAX::KERNELS::spgemm_initialize(context, this->A,
                                                        this->B, this->C_ref);
            },
            "spgemm_initialize",
            [this](KernelContext context) {
                return SMAX::KERNELS::sptrsv_initialize(context, this->A,
                                                        this->dX, this->dY);
            },
            "sptrsv_initialize",
            [this](KernelContext context) {
                return SMAX::KERNELS::sptrsm_initialize(context, this->A,
                                                        this->dX, this->dY);
            },
            "sptrsm_initialize");
        // this->timers->initialize_time->stop();
        return ret;
    }

    int apply(int A_offset, int B_offset, int C_offset) {
        // this->timers->apply_time->start();
        int ret = dispatch(
            [this, A_offset, B_offset, C_offset](KernelContext context) {
                return SMAX::KERNELS::spmv_apply(context, this->A, this->dX,
                                                 this->dY, A_offset, B_offset,
                                                 C_offset);
            },
            "spmv_apply",
            [this, A_offset, B_offset, C_offset](KernelContext context) {
                return SMAX::KERNELS::spmm_apply(context, this->A, this->dX,
                                                 this->dY, A_offset, B_offset,
                                                 C_offset);
            },
            "spmm_apply",
            [this](KernelContext context) {
                return SMAX::KERNELS::spgemv_apply(context, this->A, this->spX,
                                                   this->spY_ref);
            },
            "spgemv_apply",
            [this](KernelContext context) {
                return SMAX::KERNELS::spgemm_apply(context, this->A, this->B,
                                                   this->C_ref);
            },
            "spgemm_apply",
            [this](KernelContext context) {
                return SMAX::KERNELS::sptrsv_apply(context, this->A, this->dX,
                                                   this->dY);
            },
            "sptrsv_apply",
            [this](KernelContext context) {
                return SMAX::KERNELS::sptrsm_apply(context, this->A, this->dX,
                                                   this->dY);
            },
            "sptrsm_apply");
        // this->timers->apply_time->stop();
        return ret;
    }

    int finalize(int A_offset, int B_offset, int C_offset) {
        // this->timers->finalize_time->start();
        int ret = dispatch(
            [this, A_offset, B_offset, C_offset](KernelContext context) {
                return SMAX::KERNELS::spmv_finalize(context, this->A, this->dX,
                                                    this->dY, A_offset,
                                                    B_offset, C_offset);
            },
            "spmv_finalize",
            [this, A_offset, B_offset, C_offset](KernelContext context) {
                return SMAX::KERNELS::spmm_finalize(context, this->A, this->dX,
                                                    this->dY, A_offset,
                                                    B_offset, C_offset);
            },
            "spmm_finalize",
            [this](KernelContext context) {
                return SMAX::KERNELS::spgemv_finalize(context, this->A,
                                                      this->spX, this->spY_ref);
            },
            "spgemv_finalize",
            [this](KernelContext context) {
                return SMAX::KERNELS::spgemm_finalize(context, this->A, this->B,
                                                      this->C_ref);
            },
            "spgemm_finalize",
            [this](KernelContext context) {
                return SMAX::KERNELS::sptrsv_finalize(context, this->A,
                                                      this->dX, this->dY);
            },
            "sptrsv_finalize",
            [this](KernelContext context) {
                return SMAX::KERNELS::sptrsm_finalize(context, this->A,
                                                      this->dX, this->dY);
            },
            "sptrsm_finalize");
        // this->timers->finalize_time->stop();
        return ret;
    }

    int run(int A_offset = 0, int B_offset = 0, int C_offset = 0) {

        CHECK_ERROR(initialize(A_offset, B_offset, C_offset), "initialize");
        CHECK_ERROR(apply(A_offset, B_offset, C_offset), "apply");
        CHECK_ERROR(finalize(A_offset, B_offset, C_offset), "finalize");

        return 0;
    }

    // TODO: each kernel should have it's own timers
    // void print_timers()
    // {
    //     this->timers->total_time->stop();

    //     long double total_time = this->timers->total_time->get_wtime();
    //     long double register_A_time =
    //     this->timers->register_A_time->get_wtime(); long double
    //     register_B_time = this->timers->register_B_time->get_wtime(); long
    //     double register_C_time = this->timers->register_C_time->get_wtime();
    //     long double initialize_time =
    //     this->timers->initialize_time->get_wtime(); long double apply_time =
    //     this->timers->apply_time->get_wtime(); long double finalize_time =
    //     this->timers->finalize_time->get_wtime();

    //     int right_flush_width = 30;
    //     int left_flush_width = 25;

    //     std::cout << std::endl;
    //     std::cout << std::scientific;
    //     std::cout << std::setprecision(3);

    //     std::cout <<
    //     "+---------------------------------------------------------+" <<
    //     std::endl; std::cout << std::left << std::setw(left_flush_width) <<
    //     "Total elapsed time: " << std::right << std::setw(right_flush_width);
    //     std::cout << total_time << "[s]" << std::endl;
    //     std::cout << std::left << std::setw(left_flush_width) << "| Register
    //     Structs time: " << std::right << std::setw(right_flush_width);
    //     std::cout << register_A_time + register_A_time + register_B_time +
    //     register_C_time << "[s]" << std::endl; std::cout << std::left <<
    //     std::setw(left_flush_width) << "| Initialize time: " << std::right <<
    //     std::setw(right_flush_width); std::cout << initialize_time << "[s]"
    //     << std::endl; std::cout << std::left << std::setw(left_flush_width)
    //     << "| Apply time: " << std::right << std::setw(right_flush_width);
    //     std::cout << apply_time << "[s]" << std::endl;
    //     std::cout << std::left << std::setw(left_flush_width) << "| Finalize
    //     time: " << std::right << std::setw(right_flush_width); std::cout <<
    //     finalize_time << "[s]" << std::endl; std::cout <<
    //     "+---------------------------------------------------------+" <<
    //     std::endl; std::cout << std::endl;
    // }
};

} // namespace SMAX

#endif // SMAX_KERNEL_HPP
