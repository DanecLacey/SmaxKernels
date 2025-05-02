#include "interface.hpp"
#include "common.hpp"
#include "error_handler.hpp"

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
}

Interface::~Interface() {

    // Delete each registered kernel
    for (auto &kv : kernels)
        delete kv.second;

    delete timers;

    ErrorHandler::close_log();
}

int Interface::register_kernel(const std::string &name, KernelType type,
                               PlatformType platform, IntType int_type,
                               FloatType float_type) {

    this->kernels[name] = new Kernel(type, platform, int_type, float_type);
    this->kernels[name]->A = new SparseMatrix{};
    this->kernels[name]->B = new SparseMatrix{};
    this->kernels[name]->C = new SparseMatrix{};
    this->kernels[name]->C_ref = new SparseMatrixRef{};
    this->kernels[name]->X = new DenseMatrix{};
    this->kernels[name]->Y = new DenseMatrix{};

    return 0;
}

void Interface::print_timers() {

    this->timers->total_time->stop();

    // TODO: Accumulate time from each registered kernel
    long double total_time = this->timers->total_time->get_wtime();
    // long double register_A_time = this->timers->register_A_time->get_wtime();
    // long double register_B_time = this->timers->register_B_time->get_wtime();
    // long double register_C_time = this->timers->register_C_time->get_wtime();
    // long double initialize_time = this->timers->initialize_time->get_wtime();
    // long double apply_time = this->timers->apply_time->get_wtime();
    // long double finalize_time = this->timers->finalize_time->get_wtime();

    int right_flush_width = 30;
    int left_flush_width = 25;

    std::cout << std::endl;
    std::cout << std::scientific;
    std::cout << std::setprecision(3);

    std::cout << "+---------------------------------------------------------+"
              << std::endl;
    std::cout << std::left << std::setw(left_flush_width)
              << "Total elapsed time: " << std::right
              << std::setw(right_flush_width);
    std::cout << total_time << "[s]" << std::endl;
    // std::cout << std::left << std::setw(left_flush_width) << "| Register
    // Structs time: " << std::right << std::setw(right_flush_width); std::cout
    // << register_A_time + register_A_time + register_B_time + register_C_time
    // << "[s]" << std::endl; std::cout << std::left <<
    // std::setw(left_flush_width) << "| Initialize time: " << std::right <<
    // std::setw(right_flush_width); std::cout << initialize_time << "[s]" <<
    // std::endl; std::cout << std::left << std::setw(left_flush_width) << "|
    // Apply time: " << std::right << std::setw(right_flush_width); std::cout <<
    // apply_time << "[s]" << std::endl; std::cout << std::left <<
    // std::setw(left_flush_width) << "| Finalize time: " << std::right <<
    // std::setw(right_flush_width); std::cout << finalize_time << "[s]" <<
    // std::endl;
    std::cout << "+---------------------------------------------------------+"
              << std::endl;
    std::cout << std::endl;
}

} // namespace SMAX