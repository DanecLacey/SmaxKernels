#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#include <string>
#include <iostream>
#include <functional>

#include "common.hpp"
#include "macros.hpp"
#include "kernels/spmv.hpp"
#include "kernels/spgemm.hpp"

namespace SMAX
{
    class Interface;

    class Interface
    {
    private:
        // One function pointer and const char* for each kernel
        int dispatch(
            std::function<int(KernelContext)>, const char *,
            std::function<int(KernelContext)>, const char *);

    public:
        // Structs which are passed to the interface object
        KernelType kernel_type;
        PlatformType platform_type;
        IntType int_type;
        FloatType float_type;
        KernelContext context;

        SparseMatrix *A;
        SparseMatrix *B;
        SparseMatrix *C;
        DenseMatrix *X;
        DenseMatrix *Y;

        Timers *timers;

        // These are the only *methods* users should interact with
        Interface(KernelType, PlatformType, IntType = UINT32, FloatType = FLOAT64);
        ~Interface();

        // C-style variadic function arg list
        int register_A(...);
        int register_B(...);
        int register_C(...);

        int initialize();
        int apply();
        int finalize();

        // Combination of initialize, apply, and finalize
        int run();

        void print_timers();
    };

} // namespace SMAX

#endif // INTERFACE_HPP
