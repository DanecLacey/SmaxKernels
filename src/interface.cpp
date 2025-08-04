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

    // Initialize timers
    this->timers = new Timers{};

    // Initialize logging
    ErrorHandler::initialize_log(".smax_log");

    // Initialize utils with modifiable Interface data
    this->uc = new UtilitiesContainer();
    this->utils = new Utils(this->kernels, this->uc);
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

int Interface::get_n_levels(void) const { return this->uc->n_levels; }

int Interface::get_level_ptr_at(int idx) const {
    return this->uc->lvl_ptr[idx];
}

int Interface::register_kernel(const std::string &name, KernelType kernel_type,
                               PlatformType platform, IntType int_type,
                               FloatType float_type) {

#if !SMAX_CUDA_MODE
    if (platform == PlatformType::CUDA)
        ErrorHandler::fatal(
            "Cannot register CUDA kernel. SMAX not built with CUDA enabled.");
#endif

    std::unique_ptr<KernelContext> k_ctx = std::make_unique<KernelContext>(
        kernel_type, platform, int_type, float_type);

    // clang-format off
    switch (kernel_type) {
    case KernelType::SPMV: {
        this->kernels[name] = std::make_unique<KERNELS::SpMVKernel>(std::move(k_ctx));
        auto* spmv = dynamic_cast<KERNELS::SpMVKernel*>(this->kernels[name].get());
        spmv->args = std::make_unique<KERNELS::SPMV::Args>(this->uc);
        spmv->flags = std::make_unique<KERNELS::SPMV::Flags>();
        break;
    }
    case KernelType::SPMM: {
        this->kernels[name] = std::make_unique<KERNELS::SpMMKernel>(std::move(k_ctx));
        auto* spmm = dynamic_cast<KERNELS::SpMMKernel*>(this->kernels[name].get());
        spmm->args = std::make_unique<KERNELS::SPMM::Args>(this->uc);
        spmm->flags = std::make_unique<KERNELS::SPMM::Flags>();
        break;
    }
    case KernelType::SPGEMM: {
        this->kernels[name] = std::make_unique<KERNELS::SpGEMMKernel>(std::move(k_ctx));
        auto* spgemm = dynamic_cast<KERNELS::SpGEMMKernel*>(this->kernels[name].get());
        spgemm->args = std::make_unique<KERNELS::SPGEMM::Args>(this->uc);
        spgemm->flags = std::make_unique<KERNELS::SPGEMM::Flags>();
        break;
    }
    case KernelType::SPTRSV: {
        this->kernels[name] = std::make_unique<KERNELS::SpTRSVKernel>(std::move(k_ctx));
        auto* sptrsv = dynamic_cast<KERNELS::SpTRSVKernel*>(this->kernels[name].get());
        sptrsv->args = std::make_unique<KERNELS::SPTRSV::Args>(this->uc);
        sptrsv->flags = std::make_unique<KERNELS::SPTRSV::Flags>();
        break;
    }
    case KernelType::SPTRSM: {
        this->kernels[name] = std::make_unique<KERNELS::SpTRSMKernel>(std::move(k_ctx));
        auto* sptrsm = dynamic_cast<KERNELS::SpTRSMKernel*>(this->kernels[name].get());
        sptrsm->args = std::make_unique<KERNELS::SPTRSM::Args>(this->uc);
        sptrsm->flags = std::make_unique<KERNELS::SPTRSM::Flags>();
        break;
    }
    default:
        std::cerr << "Error: Kernel not supported\n";
        return 1;
    }
    // clang-format on

    return 0;
}

} // namespace SMAX
