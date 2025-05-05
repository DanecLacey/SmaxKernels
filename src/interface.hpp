#ifndef SMAX_INTERFACE_HPP
#define SMAX_INTERFACE_HPP

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "common.hpp"
#include "kernel.hpp"
#include "macros.hpp"

namespace SMAX {

class Interface {
  public:
    std::unordered_map<std::string, Kernel *> kernels;

    // #ifdef DEBUG_MODE
    //     Kernel *&operator[](const std::string &name);
    // #endif

    int register_kernel(const std::string &, KernelType, PlatformType,
                        IntType = UINT32, FloatType = FLOAT64);

    Timers *timers;

    Interface();
    ~Interface();

    // C-style variadic function arg list
    int register_A(...);
    int register_B(...);
    int register_C(...);

    void print_timers();
};

} // namespace SMAX

#endif // SMAX_INTERFACE_HPP
