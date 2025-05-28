#pragma once

#include "../common.hpp"
#include "utils_common.hpp"

namespace SMAX {
// clang-format off
void print_spmv_timers(const std::string &kernel_name, const Kernel *kernel_ptr, 
    const std::function<void(const std::string &, double)> &printer) {
    long double initialize_time = kernel_ptr->timers->get("initialize")->get_wtime();
    long double apply_time = kernel_ptr->timers->get("apply")->get_wtime();
    long double finalize_time = kernel_ptr->timers->get("finalize")->get_wtime();
    long double total_time = initialize_time + apply_time + finalize_time;

    printer("Total time in '" + kernel_name + "':", total_time);
    printer("| Initialize time:", initialize_time);
    printer("| Apply time:", apply_time);
    printer("| Finalize time:", finalize_time);

}

void print_spgemm_timers(const std::string &kernel_name, const Kernel *kernel_ptr, 
    const std::function<void(const std::string &, double)> &printer) {
    long double initialize_time = kernel_ptr->timers->get("initialize")->get_wtime();
    long double apply_time = kernel_ptr->timers->get("apply")->get_wtime();
    long double symbolic_phase_time = kernel_ptr->timers->get("symbolic_phase")->get_wtime();
    long double symbolic_setup_time = kernel_ptr->timers->get("Symbolic_Setup")->get_wtime();
    long double symbolic_gustavson_time = kernel_ptr->timers->get("Symbolic_Gustavson")->get_wtime();
    long double alloc_time = kernel_ptr->timers->get("Alloc_C")->get_wtime();
    long double compress_time = kernel_ptr->timers->get("Compress")->get_wtime();
    long double numerical_phase_time = kernel_ptr->timers->get("numerical_phase")->get_wtime();
    long double numerical_setup_time = kernel_ptr->timers->get("Numerical_Setup")->get_wtime();
    long double numerical_gustavson_time = kernel_ptr->timers->get("Numerical_Gustavson")->get_wtime();
    long double finalize_time = kernel_ptr->timers->get("finalize")->get_wtime();
    long double total_time = initialize_time + apply_time + finalize_time;

    printer("Total time in '" + kernel_name + "':", total_time);
    printer("| Initialize time:", initialize_time);
    printer("| Apply time:", apply_time);
    printer("| | Symbolic Phase time:", symbolic_phase_time);
    printer("| | | Setup time:", symbolic_setup_time);
    printer("| | | Gustavson time:", symbolic_gustavson_time);
    printer("| | | Alloc C time:", alloc_time);
    printer("| | | Compress time:", compress_time);
    printer("| | Numerical Phase time:", numerical_phase_time);
    printer("| | | Setup time:", numerical_setup_time);
    printer("| | | Gustavson time:", numerical_gustavson_time);
    printer("| Finalize time:", finalize_time);

}

void print_sptrsv_timers(const std::string &kernel_name, const Kernel *kernel_ptr, 
    const std::function<void(const std::string &, double)> &printer) {
    long double initialize_time = kernel_ptr->timers->get("initialize")->get_wtime();
    long double apply_time = kernel_ptr->timers->get("apply")->get_wtime();
    long double finalize_time = kernel_ptr->timers->get("finalize")->get_wtime();
    long double total_time = initialize_time + apply_time + finalize_time;

    printer("Total time in '" + kernel_name + "':", total_time);
    printer("| Initialize time:", initialize_time);
    printer("| Apply time:", apply_time);
    printer("| Finalize time:", finalize_time);

}

void Utils::print_timers() {
#ifdef USE_TIMERS
    // Compute the longest label
    int left_flush_width = 0;
    for (auto &[kernel_name, kernel_ptr] : this->kernels) {
        std::string title = "Total time in '" + kernel_name + "':";
        left_flush_width =
            std::max(left_flush_width, static_cast<int>(title.length()));
    }
    left_flush_width += 2; // Add padding
    int right_flush_width = 12;
    int total_content_width = left_flush_width + right_flush_width + 2;

    std::cout << std::scientific;
    std::cout << std::setprecision(3);

    // Helpers //
    auto print_timing = [&](const std::string &label, double time) {
    std::cout << std::left << std::setw(left_flush_width) << label
        << std::right << std::setw(right_flush_width) << time
        << " [s]\n";
    };
    
    std::string label = "SMAX Timer";
    int label_len = label.size();
    int dash_total = total_content_width - label_len;
    int dash_left = dash_total / 2;
    int dash_right = dash_total - dash_left;
    // Helpers //

    for (auto &[kernel_name, kernel_ptr] : this->kernels) {
        std::cout << '+' <<  std::string(dash_left, '-') << label << std::string(dash_right, '-') << "+\n";

        switch (kernel_ptr->k_ctx->kernel_type) {
        case KernelType::SPMV:
        case KernelType::SPMM: {
            print_spmv_timers(kernel_name, kernel_ptr.get(), print_timing);
            break;
        }
        case KernelType::SPGEMM: {
            print_spgemm_timers(kernel_name, kernel_ptr.get(), print_timing);
            break;
        }
        case KernelType::SPTRSV:
        case KernelType::SPTRSM: {
            print_sptrsv_timers(kernel_name, kernel_ptr.get(), print_timing);
            break;
        }
        default:
            std::cerr << "Error: Kernel not supported\n";
        }

        std::cout << '+' << std::string(total_content_width, '-') << "+\n";
    }

#else
    printf("USE_TIMERS is disabled in SMAX. No timers collected.\n");
#endif
}
// clang-format on
} // namespace SMAX