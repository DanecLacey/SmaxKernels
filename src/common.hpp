#ifndef SMAX_COMMON_HPP
#define SMAX_COMMON_HPP

#include "error_handler.hpp"
#include "macros.hpp"
#include "memory_utils.hpp"
#include "stopwatch.hpp"

namespace SMAX {

// Available kernels
enum KernelType { SPMV, SPGEMM, SPTSV };

// Available platforms
enum PlatformType { CPU };

// Available integer types
enum IntType { UINT16, UINT32, UINT64 };

// Available floating point types
enum FloatType { FLOAT32, FLOAT64 };

typedef struct {
    KernelType kernel_type;
    PlatformType platform_type;
    IntType int_type;
    FloatType float_type;
} KernelContext;

// For simplicity, assume all matrices are in CSR format
typedef struct {
    void *n_rows;
    void *n_cols;
    void *nnz;
    void **col;
    void **row_ptr;
    void **val;
} SparseMatrix;

// TODO
// // Available sparse matrix storage formats
// enum SparseMatrixStorageFormat
// {
//     CRS,
//     SCS
// };

typedef struct {
    void *n_rows;
    void *n_cols;
    void **val;
} DenseMatrix;

// TODO
// // Available dense matrix storage formats
// enum DenseMatrixStorageFormat
// {
//     COLWISE,
//     ROWWISE
// };

} // namespace SMAX

#endif // SMAX_COMMON_HPP
