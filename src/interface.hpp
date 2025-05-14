#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "common.hpp"
#include "kernel.hpp"
#include "macros.hpp"
#include "utils.hpp"

namespace SMAX {

class Interface {
  private:
    // DL 09.05.2025 NOTE: Populated with Utils::generate_perm. When would we
    // need multiple lvl_ptrs in the same interface?
    UtilitiesContainer *uc;

    // DL 14.05.25 NOTE: "kernels" map is made of base class pointers, but every
    // object pointed to is actually a kernel-specific sub class. This just
    // makes the interface cleaner. Kernel-specific args are still held in the
    // base class, the sub classes are just a means of avoiding switch-case
    // statements at runtime
    std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels;

  public:
    int register_kernel(const std::string &, KernelType,
                        PlatformType = PlatformType::CPU,
                        IntType = IntType::UINT32,
                        FloatType = FloatType::FLOAT64);

    Timers *timers;
    Utils *utils;

    Interface();
    ~Interface();

    Kernel *kernel(const std::string &kernel_name);

    void print_timers();
};

} // namespace SMAX
