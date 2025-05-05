#ifndef SMAX_COMMON_HPP
#define SMAX_COMMON_HPP

#include "error_handler.hpp"
#include "macros.hpp"
#include "memory_utils.hpp"
#include "stopwatch.hpp"

namespace SMAX {

// Available kernels
enum KernelType { SPMV, SPMM, SPGEMV, SPGEMM, SPTRSV, SPTRSM };

// Available platforms
enum PlatformType { CPU };

// Available integer types
enum IntType { UINT16, UINT32, UINT64 };

// Available floating point types
enum FloatType { FLOAT32, FLOAT64 };

struct KernelContext {
    KernelType kernel_type;
    PlatformType platform_type;
    IntType int_type;
    FloatType float_type;
};

// For simplicity, assume all matrices are in CSR format
// DL 5.5.25 TODO: Rename to SparseMatrixCRS
struct SparseMatrix {
    int n_rows;
    int n_cols;
    int nnz;
    void **col;
    void **row_ptr;
    void **val;

    // Default constructor
    SparseMatrix()
        : n_rows(0), n_cols(0), nnz(0), col(nullptr), row_ptr(nullptr),
          val(nullptr) {}
};

// Workaround for SPGEMM result matrix
struct SparseMatrixRef {
    int *n_rows;
    int *n_cols;
    int *nnz;
    void **col;
    void **row_ptr;
    void **val;

    // Default constructor
    SparseMatrixRef()
        : n_rows(nullptr), n_cols(nullptr), nnz(nullptr), col(nullptr),
          row_ptr(nullptr), val(nullptr) {}
};

// DL 5.5.25 TODO: Wrap up as a special case of SparseMatrixCOO
struct SparseVector {
    int n_rows; // total number of entries (implied zero + nonzero)
    int nnz;    // number of nonzeros
    void **idx; // [nnz] index of each nonzero
    void **val; // [nnz] value of each nonzero

    // Default constructor
    SparseVector() : n_rows(0), nnz(0), idx(nullptr), val(nullptr) {}
};

// Workaround
struct SparseVectorRef {
    int *n_rows;
    int *nnz;
    void **idx;
    void **val;

    // Default constructor
    SparseVectorRef()
        : n_rows(nullptr), nnz(nullptr), idx(nullptr), val(nullptr) {}
};

// TODO
// // Available sparse matrix storage formats
// enum SparseMatrixStorageFormat
// {
//     CRS,
//     SCS
// };

struct DenseMatrix {
    int n_rows;
    int n_cols;
    void **val;

    // Default constructor
    DenseMatrix() : n_rows(0), n_cols(0), val(nullptr) {}

    // Parameterized constructor
    DenseMatrix(int rows, int cols, void **val_ptr = nullptr)
        : n_rows(rows), n_cols(cols), val(val_ptr) {}
};

// TODO
// // Available dense matrix storage formats
// enum DenseMatrixStorageFormat
// {
//     COLWISE,
//     ROWWISE
// };

} // namespace SMAX

#endif // SMAX_COMMON_HPP
