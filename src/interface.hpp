#ifndef SMAX_INTERFACE_HPP
#define SMAX_INTERFACE_HPP

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
  protected:
    // DL 09.05.2025 NOTE: Populated with Utils::generate_perm. When would we
    // need multiple lvl_ptrs in the same interface?
    UtilitiesContainer *uc;

  public:
    std::unordered_map<std::string, Kernel *> kernels;

    int register_kernel(const std::string &, KernelType, PlatformType,
                        IntType = UINT32, FloatType = FLOAT64);

    Timers *timers;
    Utils *utils;

    Interface();
    ~Interface();

    void print_timers();
};

} // namespace SMAX

#endif // SMAX_INTERFACE_HPP
