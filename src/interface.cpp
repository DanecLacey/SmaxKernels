#include "interface.hpp"

#include <cstdarg>
#include <string>
#include <iostream>
#include <functional>

namespace SMAX
{

    Interface::Interface(
        KernelType _kernel_type,
        PlatformType _platform_type,
        IntType _int_type,
        FloatType _float_type) : kernel_type(_kernel_type),
                                 platform_type(_platform_type),
                                 int_type(_int_type),
                                 float_type(_float_type)
    {
        // Initialize the timers
        this->timers = new Timers{};
        init_timers(this->timers);
        this->timers->total_time->start();

        this->context = {kernel_type, platform_type, int_type, float_type};

        this->A = new SparseMatrix{};
        this->B = new SparseMatrix{};
        this->C = new SparseMatrix{};
        this->x = new DenseVector{};
        this->y = new DenseVector{};
    }

    Interface::~Interface()
    {
        delete timers;
        delete A;
        delete B;
        delete C;
        delete x;
        delete y;
    }

    int Interface::register_A(...)
    {
        this->timers->register_A_time->start();

        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type)
        {
        case SPMV:
        {
            int ret = spmv_register_A(this->A, user_args);
            va_end(user_args);
            this->timers->register_A_time->stop();
            return ret;
        }
        case SPGEMM:
        {
            int ret = spgemm_register_A(this->A, user_args);
            va_end(user_args);
            this->timers->register_A_time->stop();
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        this->timers->register_A_time->stop();
        return 0;
    }

    int Interface::register_B(...)
    {
        this->timers->register_B_time->start();
        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type)
        {
        case SPMV:
        {
            int ret = spmv_register_B(this->x, user_args);
            va_end(user_args);
            this->timers->register_B_time->stop();
            return ret;
        }
        case SPGEMM:
        {
            int ret = spgemm_register_B(this->B, user_args);
            va_end(user_args);
            this->timers->register_B_time->stop();
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        this->timers->register_B_time->stop();
        return 0;
    }

    int Interface::register_C(...)
    {
        this->timers->register_C_time->start();
        va_list user_args;
        va_start(user_args, this);

        switch (this->context.kernel_type)
        {
        case SPMV:
        {
            int ret = spmv_register_C(this->y, user_args);
            va_end(user_args);
            this->timers->register_C_time->stop();
            return ret;
        }
        case SPGEMM:
        {
            int ret = spgemm_register_C(this->C, user_args);
            va_end(user_args);
            this->timers->register_C_time->stop();
            return ret;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }

        this->timers->register_C_time->stop();
        return 0;
    }

    int Interface::dispatch(
        std::function<int(KernelContext)> func_spmv, const char *label_spmv,
        std::function<int(KernelContext)> func_spgemm, const char *label_spgemm)
    {
        switch (this->context.kernel_type)
        {
        case SMAX::SPMV:
            CHECK_ERROR(func_spmv(this->context), label_spmv);
            break;
        case SMAX::SPGEMM:
            CHECK_ERROR(func_spgemm(this->context), label_spgemm);
            break;
        default:
            std::cerr << "Error: Kernel not supported\n";
            return 1;
        }
        return 0;
    }

    int Interface::initialize()
    {
        this->timers->initialize_time->start();
        int ret = dispatch(
            [this](auto context)
            { return spmv_initialize(context, this->A, this->x, this->y); }, "spmv_initialize",
            [this](auto context)
            { return spgemm_initialize(context, this->A, this->B, this->C); }, "spgemm_initialize");
        this->timers->initialize_time->stop();
        return ret;
    }

    int Interface::apply()
    {
        this->timers->apply_time->start();
        int ret = dispatch(
            [this](auto context)
            { return spmv_apply(context, this->A, this->x, this->y); }, "spmv_apply",
            [this](auto context)
            { return spgemm_apply(context, this->A, this->B, this->C); }, "spgemm_apply");
        this->timers->apply_time->stop();
        return ret;
    }

    int Interface::finalize()
    {
        this->timers->finalize_time->start();
        int ret = dispatch(
            [this](auto context)
            { return spmv_finalize(context, this->A, this->x, this->y); }, "spmv_finalize",
            [this](auto context)
            { return spgemm_finalize(context, this->A, this->B, this->C); }, "spgemm_finalize");
        this->timers->finalize_time->stop();
        return ret;
    }

    int Interface::run(

    )
    {
        CHECK_ERROR(initialize(), "initialize");
        CHECK_ERROR(apply(), "apply");
        CHECK_ERROR(finalize(), "finalize");

        return 0;
    }

    void Interface::print_timers()
    {
        this->timers->total_time->stop();

        long double total_time = this->timers->total_time->get_wtime();
        long double register_A_time = this->timers->register_A_time->get_wtime();
        long double register_B_time = this->timers->register_B_time->get_wtime();
        long double register_C_time = this->timers->register_C_time->get_wtime();
        long double initialize_time = this->timers->initialize_time->get_wtime();
        long double apply_time = this->timers->apply_time->get_wtime();
        long double finalize_time = this->timers->finalize_time->get_wtime();

        int right_flush_width = 30;
        int left_flush_width = 25;

        std::cout << std::endl;
        std::cout << std::scientific;
        std::cout << std::setprecision(3);

        std::cout << "+---------------------------------------------------------+" << std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "Total elapsed time: " << std::right << std::setw(right_flush_width);
        std::cout << total_time << "[s]" << std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| Register Structs time: " << std::right << std::setw(right_flush_width);
        std::cout << register_A_time + register_A_time + register_B_time + register_C_time << "[s]" << std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| Initialize time: " << std::right << std::setw(right_flush_width);
        std::cout << initialize_time << "[s]" << std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| Apply time: " << std::right << std::setw(right_flush_width);
        std::cout << apply_time << "[s]" << std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| Finalize time: " << std::right << std::setw(right_flush_width);
        std::cout << finalize_time << "[s]" << std::endl;
        std::cout << "+---------------------------------------------------------+" << std::endl;
        std::cout << std::endl;
    }

} // namespace SMAX