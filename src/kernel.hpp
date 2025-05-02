#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "common.hpp"
#include "kernels/spgemm.hpp"
#include "kernels/spmv.hpp"
#include "kernels/sptsv.hpp"

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
    DenseMatrix *X;
    DenseMatrix *Y;
    KernelContext context;

    Kernel(KernelType _kernel_type, PlatformType _platform, IntType _int_type,
           FloatType _float_type)
        : kernel_type(_kernel_type), platform(_platform), int_type(_int_type),
          float_type(_float_type) {

        this->context = {kernel_type, platform, int_type, float_type};
    }

    ~Kernel() {
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_ref;
        delete[] X;
        delete[] Y;
    }

    // Register A, B, C matrices and x, y vectors
    // These functions are variadic and take a variable number of arguments
    int register_A(...) {
        // this->timers->register_A_time->start();

        va_list user_args;
        va_start(user_args, this);

        switch (this->kernel_type) {
        case SPMV: {
            int ret = SMAX::KERNELS::spmv_register_A(this->A, user_args);
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
        case SPTSV: {
            int ret = SMAX::KERNELS::sptsv_register_A(this->A, user_args);
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

        switch (this->kernel_type) {
        case SPMV: {
            int ret = SMAX::KERNELS::spmv_register_B(this->X, user_args);
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
        case SPTSV: {
            int ret = SMAX::KERNELS::sptsv_register_B(this->X, user_args);
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

        switch (this->kernel_type) {
        case SPMV: {
            int ret = SMAX::KERNELS::spmv_register_C(this->Y, user_args);
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
        case SPTSV: {
            int ret = SMAX::KERNELS::sptsv_register_C(this->Y, user_args);
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

    int dispatch(std::function<int(KernelContext)> func_spmv,
                 const char *label_spmv,
                 std::function<int(KernelContext)> func_spgemm,
                 const char *label_spgemm,
                 std::function<int(KernelContext)> func_sptsv,
                 const char *label_sptsv) {
        switch (this->kernel_type) {
        case SMAX::SPMV:
            CHECK_ERROR(func_spmv(this->context), label_spmv);
            break;
        case SMAX::SPGEMM:
            CHECK_ERROR(func_spgemm(this->context), label_spgemm);
            break;
        case SMAX::SPTSV:
            CHECK_ERROR(func_sptsv(this->context), label_sptsv);
            break;
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }
        return 0;
    }

    int initialize() {
        // this->timers->initialize_time->start();
        int ret = dispatch(
            [this](auto context) {
                return SMAX::KERNELS::spmv_initialize(context, this->A, this->X,
                                                      this->Y);
            },
            "spmv_initialize",
            [this](auto context) {
                return SMAX::KERNELS::spgemm_initialize(context, this->A,
                                                        this->B, this->C_ref);
            },
            "spgemm_initialize",
            [this](auto context) {
                return SMAX::KERNELS::sptsv_initialize(context, this->A,
                                                       this->X, this->Y);
            },
            "sptsv_initialize");
        // this->timers->initialize_time->stop();
        return ret;
    }

    int apply() {
        // this->timers->apply_time->start();
        int ret = dispatch(
            [this](auto context) {
                return SMAX::KERNELS::spmv_apply(context, this->A, this->X,
                                                 this->Y);
            },
            "spmv_apply",
            [this](auto context) {
                return SMAX::KERNELS::spgemm_apply(context, this->A, this->B,
                                                   this->C_ref);
            },
            "spgemm_apply",
            [this](auto context) {
                return SMAX::KERNELS::sptsv_apply(context, this->A, this->X,
                                                  this->Y);
            },
            "sptsv_apply");
        // this->timers->apply_time->stop();
        return ret;
    }

    int finalize() {
        // this->timers->finalize_time->start();
        int ret = dispatch(
            [this](auto context) {
                return SMAX::KERNELS::spmv_finalize(context, this->A, this->X,
                                                    this->Y);
            },
            "spmv_finalize",
            [this](auto context) {
                return SMAX::KERNELS::spgemm_finalize(context, this->A, this->B,
                                                      this->C_ref);
            },
            "spgemm_finalize",
            [this](auto context) {
                return SMAX::KERNELS::sptsv_finalize(context, this->A, this->X,
                                                     this->Y);
            },
            "sptsv_finalize");
        // this->timers->finalize_time->stop();
        return ret;
    }

    int run(

    ) {
        CHECK_ERROR(initialize(), "initialize");
        CHECK_ERROR(apply(), "apply");
        CHECK_ERROR(finalize(), "finalize");

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

#endif // KERNEL_HPP
