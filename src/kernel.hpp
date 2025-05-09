#ifndef SMAX_KERNEL_HPP
#define SMAX_KERNEL_HPP

#include "common.hpp"
#include "kernels/spgemm.hpp"
#include "kernels/spgemm/spgemm_common.hpp"
#include "kernels/spgemv.hpp"
#include "kernels/spgemv/spgemv_common.hpp"
#include "kernels/spmm.hpp"
#include "kernels/spmm/spmm_common.hpp"
#include "kernels/spmv.hpp"
#include "kernels/spmv/spmv_common.hpp"
#include "kernels/sptrsm.hpp"
#include "kernels/sptrsm/sptrsm_common.hpp"
#include "kernels/sptrsv.hpp"
#include "kernels/sptrsv/sptrsv_common.hpp"

#include <cstdarg>

namespace SMAX {

class Kernel {
  private:
    KernelType kernel_type;
    PlatformType platform;
    IntType int_type;
    FloatType float_type;

  public:
    // DL 06.05.2025 TODO: Do these need to be public? Or am I being lazy?
    KernelContext context;
    UtilitiesContainer *uc;

    // DL 07.05.2025 NOTE: Only the chosen kernel has args struct populated
    // A bit wasteful, but cleaner interface
    KERNELS::SPMV::Args *spmv_args;
    KERNELS::SPMV::Flags *spmv_flags;
    KERNELS::SPMM::Args *spmm_args;
    KERNELS::SPMM::Flags *spmm_flags;
    KERNELS::SPGEMV::Args *spgemv_args;
    KERNELS::SPGEMV::Flags *spgemv_flags;
    KERNELS::SPGEMM::Args *spgemm_args;
    KERNELS::SPGEMM::Flags *spgemm_flags;
    KERNELS::SPTRSV::Args *sptrsv_args;
    KERNELS::SPTRSV::Flags *sptrsv_flags;
    KERNELS::SPTRSM::Args *sptrsm_args;
    KERNELS::SPTRSM::Flags *sptrsm_flags;

    Kernel(KernelType kernel_type, PlatformType platform, IntType int_type,
           FloatType float_type) {

        this->context =
            KernelContext{kernel_type, platform, int_type, float_type};
    }

    ~Kernel() {
        switch (context.kernel_type) {
        case SPMV: {
            delete spmv_args;
            delete spmv_flags;
            break;
        }
        case SPMM: {
            delete spmm_args;
            delete spmm_flags;
            break;
        }
        case SPGEMV: {
            delete spgemv_args;
            delete spgemv_flags;
            break;
        }
        case SPGEMM: {
            delete spgemm_args;
            delete spgemm_flags;
            break;
        }
        case SPTRSV: {
            delete sptrsv_args;
            delete sptrsv_flags;
            break;
        }
        case SPTRSM: {
            delete sptrsm_args;
            delete sptrsm_flags;
            break;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
        }
    }

    // Flag setters
    void set_perm(bool flag) { this->sptrsv_flags->mat_permuted = flag; }

    // C-style variadic function arg list
    int register_A(...) {
        // this->timers->register_A_time->start();

        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type) {
        case SPMV: {
            int ret = KERNELS::spmv_register_A(this->spmv_args->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPMM: {
            int ret = KERNELS::spmm_register_A(this->spmm_args->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPGEMV: {
            int ret =
                KERNELS::spgemv_register_A(this->spgemv_args->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPGEMM: {
            int ret =
                KERNELS::spgemm_register_A(this->spgemm_args->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPTRSV: {
            int ret =
                KERNELS::sptrsv_register_A(this->sptrsv_args->A, user_args);
            va_end(user_args);
            // this->timers->register_A_time->stop();
            return ret;
        }
        case SPTRSM: {
            int ret =
                KERNELS::sptrsm_register_A(this->sptrsm_args->A, user_args);
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

    // C-style variadic function arg list
    int register_B(...) {
        // this->timers->register_B_time->start();
        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type) {
        case SPMV: {
            int ret = KERNELS::spmv_register_B(this->spmv_args->x, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPMM: {
            int ret = KERNELS::spmm_register_B(this->spmm_args->X, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPGEMV: {
            int ret =
                KERNELS::spgemv_register_B(this->spgemv_args->x, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPGEMM: {
            int ret =
                KERNELS::spgemm_register_B(this->spgemm_args->B, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPTRSV: {
            int ret =
                KERNELS::sptrsv_register_B(this->sptrsv_args->x, user_args);
            va_end(user_args);
            // this->timers->register_B_time->stop();
            return ret;
        }
        case SPTRSM: {
            int ret =
                KERNELS::sptrsm_register_B(this->sptrsm_args->X, user_args);
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

    // C-style variadic function arg list
    int register_C(...) {
        // this->timers->register_C_time->start();
        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type) {
        case SPMV: {
            int ret = KERNELS::spmv_register_C(this->spmv_args->y, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPMM: {
            int ret = KERNELS::spmm_register_C(this->spmm_args->Y, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPGEMV: {
            int ret =
                KERNELS::spgemv_register_C(this->spgemv_args->y, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPGEMM: {
            int ret =
                KERNELS::spgemm_register_C(this->spgemm_args->C, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPTRSV: {
            int ret =
                KERNELS::sptrsv_register_C(this->sptrsv_args->y, user_args);
            va_end(user_args);
            // this->timers->register_C_time->stop();
            return ret;
        }
        case SPTRSM: {
            int ret =
                KERNELS::sptrsm_register_C(this->sptrsm_args->Y, user_args);
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

    int dispatch(std::function<int(KernelContext, KERNELS::SPMV::Args *,
                                   KERNELS::SPMV::Flags *)>
                     func_spmv,
                 const char *label_spmv,
                 std::function<int(KernelContext, KERNELS::SPMM::Args *,
                                   KERNELS::SPMM::Flags *)>
                     func_spmm,
                 const char *label_spmm,
                 std::function<int(KernelContext, KERNELS::SPGEMV::Args *,
                                   KERNELS::SPGEMV::Flags *)>
                     func_spgemv,
                 const char *label_spgemv,
                 std::function<int(KernelContext, KERNELS::SPGEMM::Args *,
                                   KERNELS::SPGEMM::Flags *)>
                     func_spgemm,
                 const char *label_spgemm,
                 std::function<int(KernelContext, KERNELS::SPTRSV::Args *,
                                   KERNELS::SPTRSV::Flags *)>
                     func_sptrsv,
                 const char *label_sptrsv,
                 std::function<int(KernelContext, KERNELS::SPTRSM::Args *,
                                   KERNELS::SPTRSM::Flags *)>
                     func_sptrsm,
                 const char *label_sptrsm) {
        switch (this->context.kernel_type) {
        case SPMV:
            CHECK_ERROR(
                func_spmv(this->context, this->spmv_args, this->spmv_flags),
                label_spmv);
            break;
        case SPMM:
            CHECK_ERROR(
                func_spmm(this->context, this->spmm_args, this->spmm_flags),
                label_spmm);
            break;
        case SPGEMV:
            CHECK_ERROR(func_spgemv(this->context, this->spgemv_args,
                                    this->spgemv_flags),
                        label_spgemv);
            break;
        case SPGEMM:
            CHECK_ERROR(func_spgemm(this->context, this->spgemm_args,
                                    this->spgemm_flags),
                        label_spgemm);
            break;
        case SPTRSV:
            CHECK_ERROR(func_sptrsv(this->context, this->sptrsv_args,
                                    this->sptrsv_flags),
                        label_sptrsv);
            break;
        case SPTRSM:
            CHECK_ERROR(func_sptrsm(this->context, this->sptrsm_args,
                                    this->sptrsm_flags),
                        label_sptrsm);
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
            [A_offset, B_offset, C_offset](KernelContext context,
                                           KERNELS::SPMV::Args *spmv_args,
                                           KERNELS::SPMV::Flags *spmv_flags) {
                return KERNELS::spmv_initialize(context, spmv_args, spmv_flags,
                                                A_offset, B_offset, C_offset);
            },
            "spmv_initialize",
            [A_offset, B_offset, C_offset](KernelContext context,
                                           KERNELS::SPMM::Args *spmm_args,
                                           KERNELS::SPMM::Flags *spmm_flags) {
                return KERNELS::spmm_initialize(context, spmm_args, spmm_flags,
                                                A_offset, B_offset, C_offset);
            },
            "spmm_initialize",
            [](KernelContext context, KERNELS::SPGEMV::Args *spgemv_args,
               KERNELS::SPGEMV::Flags *spgemv_flags) {
                return KERNELS::spgemv_initialize(context, spgemv_args,
                                                  spgemv_flags);
            },
            "spgemv_initialize",
            [this](KernelContext context, KERNELS::SPGEMM::Args *spgemm_args,
                   KERNELS::SPGEMM::Flags *spgemm_flags) {
                return KERNELS::spgemm_initialize(context, spgemm_args,
                                                  spgemm_flags);
            },
            "spgemm_initialize",
            [this](KernelContext context, KERNELS::SPTRSV::Args *sptrsv_args,
                   KERNELS::SPTRSV::Flags *sptrsv_flags) {
                return KERNELS::sptrsv_initialize(context, sptrsv_args,
                                                  sptrsv_flags);
            },
            "sptrsv_initialize",
            [this](KernelContext context, KERNELS::SPTRSM::Args *sptrsm_args,
                   KERNELS::SPTRSM::Flags *sptrsm_flags) {
                return KERNELS::sptrsm_initialize(context, sptrsm_args,
                                                  sptrsm_flags);
            },
            "sptrsm_initialize");
        // this->timers->initialize_time->stop();
        return ret;
    }

    int apply(int A_offset, int B_offset, int C_offset) {
        // this->timers->apply_time->start();
        int ret = dispatch(
            [A_offset, B_offset, C_offset](KernelContext context,
                                           KERNELS::SPMV::Args *spmv_args,
                                           KERNELS::SPMV::Flags *spmv_flags) {
                return KERNELS::spmv_apply(context, spmv_args, spmv_flags,
                                           A_offset, B_offset, C_offset);
            },
            "spmv_apply",
            [A_offset, B_offset, C_offset](KernelContext context,
                                           KERNELS::SPMM::Args *spmm_args,
                                           KERNELS::SPMM::Flags *spmm_flags) {
                return KERNELS::spmm_apply(context, spmm_args, spmm_flags,
                                           A_offset, B_offset, C_offset);
            },
            "spmm_apply",
            [this](KernelContext context, KERNELS::SPGEMV::Args *spgemv_args,
                   KERNELS::SPGEMV::Flags *spgemv_flags) {
                return KERNELS::spgemv_apply(context, spgemv_args,
                                             spgemv_flags);
            },
            "spgemv_apply",
            [this](KernelContext context, KERNELS::SPGEMM::Args *spgemm_args,
                   KERNELS::SPGEMM::Flags *spgemm_flags) {
                return KERNELS::spgemm_apply(context, spgemm_args,
                                             spgemm_flags);
            },
            "spgemm_apply",
            [this](KernelContext context, KERNELS::SPTRSV::Args *sptrsv_args,
                   KERNELS::SPTRSV::Flags *sptrsv_flags) {
                return KERNELS::sptrsv_apply(context, sptrsv_args,
                                             sptrsv_flags);
            },
            "sptrsv_apply",
            [this](KernelContext context, KERNELS::SPTRSM::Args *sptrsm_args,
                   KERNELS::SPTRSM::Flags *sptrsm_flags) {
                return KERNELS::sptrsm_apply(context, sptrsm_args,
                                             sptrsm_flags);
            },
            "sptrsm_apply");
        // this->timers->apply_time->stop();
        return ret;
    }

    int finalize(int A_offset, int B_offset, int C_offset) {
        // this->timers->finalize_time->start();
        int ret = dispatch(
            [A_offset, B_offset, C_offset](KernelContext context,
                                           KERNELS::SPMV::Args *spmv_args,
                                           KERNELS::SPMV::Flags *spmv_flags) {
                return KERNELS::spmv_finalize(context, spmv_args, spmv_flags,
                                              A_offset, B_offset, C_offset);
            },
            "spmv_finalize",
            [this, A_offset, B_offset,
             C_offset](KernelContext context, KERNELS::SPMM::Args *spmm_args,
                       KERNELS::SPMM::Flags *spmm_flags) {
                return KERNELS::spmm_finalize(context, spmm_args, spmm_flags,
                                              A_offset, B_offset, C_offset);
            },
            "spmm_finalize",
            [this](KernelContext context, KERNELS::SPGEMV::Args *spgemv_args,
                   KERNELS::SPGEMV::Flags *spgemv_flags) {
                return KERNELS::spgemv_finalize(context, spgemv_args,
                                                spgemv_flags);
            },
            "spgemv_finalize",
            [this](KernelContext context, KERNELS::SPGEMM::Args *spgemm_args,
                   KERNELS::SPGEMM::Flags *spgemm_flags) {
                return KERNELS::spgemm_finalize(context, spgemm_args,
                                                spgemm_flags);
            },
            "spgemm_finalize",
            [this](KernelContext context, KERNELS::SPTRSV::Args *sptrsv_args,
                   KERNELS::SPTRSV::Flags *sptrsv_flags) {
                return KERNELS::sptrsv_finalize(context, sptrsv_args,
                                                sptrsv_flags);
            },
            "sptrsv_finalize",
            [this](KernelContext context, KERNELS::SPTRSM::Args *sptrsm_args,
                   KERNELS::SPTRSM::Flags *sptrsm_flags) {
                return KERNELS::sptrsm_finalize(context, sptrsm_args,
                                                sptrsm_flags);
            },
            "sptrsm_finalize");
        // this->timers->finalize_time->stop();
        return ret;
    }

    int run(int A_offset = 0, int B_offset = 0, int C_offset = 0) {
        // DL 09.05.2025 NOTE: Since only a single memory address is registered,
        // the offsets are to access other parts of memory at runtime
        // (e.g. register A to kernel, but A[i*n_rows] used for computation)
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
    //     register_B_time = this->timers->register_B_time->get_wtime();
    //     long double register_C_time =
    //     this->timers->register_C_time->get_wtime(); long double
    //     initialize_time = this->timers->initialize_time->get_wtime();
    //     long double apply_time = this->timers->apply_time->get_wtime();
    //     long double finalize_time =
    //     this->timers->finalize_time->get_wtime();

    //     int right_flush_width = 30;
    //     int left_flush_width = 25;

    //     std::cout << std::endl;
    //     std::cout << std::scientific;
    //     std::cout << std::setprecision(3);

    //     std::cout <<
    //     "+---------------------------------------------------------+" <<
    //     std::endl; std::cout << std::left << std::setw(left_flush_width)
    //     << "Total elapsed time: " << std::right <<
    //     std::setw(right_flush_width); std::cout << total_time << "[s]" <<
    //     std::endl; std::cout << std::left << std::setw(left_flush_width)
    //     << "| Register Structs time: " << std::right <<
    //     std::setw(right_flush_width); std::cout << register_A_time +
    //     register_A_time + register_B_time + register_C_time << "[s]" <<
    //     std::endl; std::cout << std::left << std::setw(left_flush_width)
    //     << "| Initialize time: " << std::right <<
    //     std::setw(right_flush_width); std::cout << initialize_time <<
    //     "[s]"
    //     << std::endl; std::cout << std::left <<
    //     std::setw(left_flush_width)
    //     << "| Apply time: " << std::right <<
    //     std::setw(right_flush_width); std::cout << apply_time << "[s]" <<
    //     std::endl; std::cout << std::left << std::setw(left_flush_width)
    //     << "| Finalize time: " << std::right <<
    //     std::setw(right_flush_width); std::cout << finalize_time << "[s]"
    //     << std::endl; std::cout <<
    //     "+---------------------------------------------------------+" <<
    //     std::endl; std::cout << std::endl;
    // }
};

} // namespace SMAX

#endif // SMAX_KERNEL_HPP
