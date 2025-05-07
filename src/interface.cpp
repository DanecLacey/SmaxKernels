#include "interface.hpp"
#include "common.hpp"
#include "error_handler.hpp"
#include "utils.hpp"

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

    // Initialize utils
    this->utils = new Utils{};
}

Interface::~Interface() {

    // Delete each registered kernel
    for (auto &kv : kernels)
        delete kv.second;

    delete this->timers;
    delete this->utils;

    ErrorHandler::close_log();
}

// #ifdef DEBUG_MODE
// Kernel *&Interface::operator[](const std::string &name) {
//     auto it = kernels.find(name);
//     if (it == kernels.end()) {
//         // Log an error internally and prevent segmentation fault
//         std::cout << "SMAX Error: Kernel \"" << name
//                   << "\" not found. Returning null pointer." << std::endl;

//         // Return a reference to a null pointer or a default Kernel pointer
//         static Kernel *nullKernel = nullptr; // static ensures it stays valid
//                                              // throughout program execution
//         return nullKernel;
//     }
//     return it->second;
// }
// #endif

int Interface::register_kernel(const std::string &name, KernelType kernel_type,
                               PlatformType platform, IntType int_type,
                               FloatType float_type) {

    // DL 07.05.2025 TODO: Kernel string validation
    // IF_DEBUG(if (this->kernels.count(name)) {
    //     std::cerr << "SMAX Warning: Kernel \"" << name
    //               << "\" already exists. Overwriting.\n";
    //     delete this->kernels[name];
    // });

    this->kernels[name] =
        new Kernel(kernel_type, platform, int_type, float_type);

    switch (kernel_type) {
    case SPMV: {
        this->kernels[name]->spmv_args = new KERNELS::SPMV::Args();
        this->kernels[name]->spmv_flags = new KERNELS::SPMV::Flags();
        break;
    }
    case SPMM: {
        this->kernels[name]->spmm_args = new KERNELS::SPMM::Args();
        this->kernels[name]->spmm_flags = new KERNELS::SPMM::Flags();
        break;
    }
    case SPGEMV: {
        this->kernels[name]->spgemv_args = new KERNELS::SPGEMV::Args();
        this->kernels[name]->spgemv_flags = new KERNELS::SPGEMV::Flags();
        break;
    }
    case SPGEMM: {
        this->kernels[name]->spgemm_args = new KERNELS::SPGEMM::Args();
        this->kernels[name]->spgemm_flags = new KERNELS::SPGEMM::Flags();
        break;
    }
    case SPTRSV: {
        this->kernels[name]->sptrsv_args = new KERNELS::SPTRSV::Args();
        this->kernels[name]->sptrsv_flags = new KERNELS::SPTRSV::Flags();
        break;
    }
    case SPTRSM: {
        this->kernels[name]->sptrsm_args = new KERNELS::SPTRSM::Args();
        this->kernels[name]->sptrsm_flags = new KERNELS::SPTRSM::Flags();
        break;
    }
    default:
        std::cerr << "Error: Kernel not supported\n";
        return 1;
    }

    // if (kernel_type == SPMV) {
    //     this->kernels[name]->spmv_args = new SpMVArgs();
    // } else {
    //     this->kernels[name]->A = new SparseMatrix();
    //     this->kernels[name]->B = new SparseMatrix();
    //     this->kernels[name]->C = new SparseMatrix();
    //     this->kernels[name]->C_ref = new SparseMatrixRef(); // SMAX will
    //     resize this->kernels[name]->dX = new DenseMatrix();
    //     this->kernels[name]->dY = new DenseMatrix();
    //     this->kernels[name]->spX = new SparseVector();
    //     this->kernels[name]->spY = new SparseVector();
    //     this->kernels[name]->spY_ref =
    //         new SparseVectorRef(); // SMAX will resize
    // }

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