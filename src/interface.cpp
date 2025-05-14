#include "interface.hpp"
#include "common.hpp"
#include "error_handler.hpp"
#include "utils.hpp"

#include "kernels/spgemm.hpp"
#include "kernels/spmm.hpp"
#include "kernels/spmv.hpp"
#include "kernels/sptrsm.hpp"
#include "kernels/sptrsv.hpp"

#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

namespace SMAX {

Interface::Interface() {

    // Initialize interface timers
    this->timers = new Timers{};
    init_timers(this->timers);
    this->timers->total_time->start();

    // Initialize logging
    ErrorHandler::initialize_log(".smax_log");

    // Initialize utils with modifiable Interface data
    this->uc = new UtilitiesContainer();
    this->utils = new Utils(this->uc);
}

Interface::~Interface() {

    delete this->timers;
    delete this->uc;
    delete this->utils;

    ErrorHandler::close_log();
}

Kernel *Interface::kernel(const std::string &kernel_name) {
    try {
        // Get raw pointer from unique_ptr
        return this->kernels.at(kernel_name).get();
    } catch (const std::out_of_range &) {
        ErrorHandler::kernel_dne(kernel_name);
        return nullptr; // Return nullptr to avoid undefined behavior
    }
}

int Interface::register_kernel(const std::string &name, KernelType kernel_type,
                               PlatformType platform, IntType int_type,
                               FloatType float_type) {

    std::unique_ptr<KernelContext> k_ctx = std::make_unique<KernelContext>(
        kernel_type, platform, int_type, float_type);

    // clang-format off
    switch (kernel_type) {
    case KernelType::SPMV: {
        this->kernels[name] = std::make_unique<KERNELS::SpMVKernel>(std::move(k_ctx));
        this->kernels[name]->spmv_args = std::make_unique<KERNELS::SPMV::Args>(this->uc);
        this->kernels[name]->spmv_flags = std::make_unique<KERNELS::SPMV::Flags>();
        break;
    }
    case KernelType::SPMM: {
        this->kernels[name] = std::make_unique<KERNELS::SpMMKernel>(std::move(k_ctx));
        this->kernels[name]->spmm_args = std::make_unique<KERNELS::SPMM::Args>(this->uc);
        this->kernels[name]->spmm_flags = std::make_unique<KERNELS::SPMM::Flags>();
        break;
    }
    case KernelType::SPGEMM: {
        this->kernels[name] = std::make_unique<KERNELS::SpGEMMKernel>(std::move(k_ctx));
        this->kernels[name]->spgemm_args = std::make_unique<KERNELS::SPGEMM::Args>(this->uc);
        this->kernels[name]->spgemm_flags = std::make_unique<KERNELS::SPGEMM::Flags>();
        break;
    }
    case KernelType::SPTRSV: {
        this->kernels[name] = std::make_unique<KERNELS::SpTRSVKernel>(std::move(k_ctx));
        this->kernels[name]->sptrsv_args = std::make_unique<KERNELS::SPTRSV::Args>(this->uc);
        this->kernels[name]->sptrsv_flags = std::make_unique<KERNELS::SPTRSV::Flags>();
        break;
    }
    case KernelType::SPTRSM: {
        this->kernels[name] = std::make_unique<KERNELS::SpTRSMKernel>(std::move(k_ctx));
        this->kernels[name]->sptrsm_args = std::make_unique<KERNELS::SPTRSM::Args>(this->uc);
        this->kernels[name]->sptrsm_flags = std::make_unique<KERNELS::SPTRSM::Flags>();
        break;
    }
    default:
        std::cerr << "Error: Kernel not supported\n";
        return 1;
    }
    // clang-format on

    return 0;
}

void Interface::print_timers() {};
// TODO: Move to utils

//     this->timers->total_time->stop();

//     // TODO: Accumulate time from each registered kernel
//     long double total_time = this->timers->total_time->get_wtime();
//     // long double register_A_time =
//     this->timers->register_A_time->get_wtime();
//     // long double register_B_time =
//     this->timers->register_B_time->get_wtime();
//     // long double register_C_time =
//     this->timers->register_C_time->get_wtime();
//     // long double initialize_time =
//     this->timers->initialize_time->get_wtime();
//     // long double apply_time = this->timers->apply_time->get_wtime();
//     // long double finalize_time =
//     this->timers->finalize_time->get_wtime();

//     int right_flush_width = 30;
//     int left_flush_width = 25;

//     std::cout << std::endl;
//     std::cout << std::scientific;
//     std::cout << std::setprecision(3);

//     std::cout <<
//     "+---------------------------------------------------------+"
//               << std::endl;
//     std::cout << std::left << std::setw(left_flush_width)
//               << "Total elapsed time: " << std::right
//               << std::setw(right_flush_width);
//     std::cout << total_time << "[s]" << std::endl;
//     // std::cout << std::left << std::setw(left_flush_width) << "|
//     Register
//     // Structs time: " << std::right << std::setw(right_flush_width);
//     std::cout
//     // << register_A_time + register_A_time + register_B_time +
//     register_C_time
//     // << "[s]" << std::endl; std::cout << std::left <<
//     // std::setw(left_flush_width) << "| Initialize time: " << std::right
//     <<
//     // std::setw(right_flush_width); std::cout << initialize_time <<
//     "[s]" <<
//     // std::endl; std::cout << std::left << std::setw(left_flush_width)
//     << "|
//     // Apply time: " << std::right << std::setw(right_flush_width);
//     std::cout
//     <<
//     // apply_time << "[s]" << std::endl; std::cout << std::left <<
//     // std::setw(left_flush_width) << "| Finalize time: " << std::right
//     <<
//     // std::setw(right_flush_width); std::cout << finalize_time << "[s]"
//     <<
//     // std::endl;
//     std::cout <<
//     "+---------------------------------------------------------+"
//               << std::endl;
//     std::cout << std::endl;
// }

} // namespace SMAX