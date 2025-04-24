#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <iomanip>

#include "memory_utils.hpp"
#include "stopwatch.hpp"

namespace SMAX
{
    // Available kernels
    enum KernelType
    {
        SPMV,
        SPGEMM
    };

    // Available platforms
    enum PlatformType
    {
        CPU
    };

    // Available integer types
    enum IntType
    {
        UINT16,
        UINT32,
        UINT64
    };

    // Available floating point types
    enum FloatType
    {
        FLOAT32,
        FLOAT64
    };

    typedef struct
    {
        KernelType kernel_type;
        PlatformType platform_type;
        IntType int_type;
        FloatType float_type;
    } KernelContext;

    // For simplicity, assume all matrices are in CSR format
    typedef struct
    {
        void *n_rows;
        void *n_cols;
        void *nnz;
        void **col;
        void **row_ptr;
        void **val;
    } SparseMatrix;

    // For simplicity, assume all vectors are dense
    typedef struct
    {
        void *n_rows;
        void **val;
    } DenseVector;

} // namespace SMAX

#endif // COMMON_HPP
