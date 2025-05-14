#pragma once

#include "error_handler.hpp"
#include "macros.hpp"
#include "memory_utils.hpp"
#include "stopwatch.hpp"

namespace SMAX {

// Available kernels
enum class KernelType { SPMV, SPMM, SPGEMV, SPGEMM, SPTRSV, SPTRSM };

// Available platforms
enum class PlatformType { CPU };

// Available integer types
enum class IntType { UINT16, UINT32, UINT64 };

// Available floating point types
enum class FloatType { FLOAT32, FLOAT64 };

struct FlagType {
    bool is_lvl_ptr_collected = false;
    bool is_mat_permuted = false;
};

struct KernelContext {
    KernelType kernel_type;
    PlatformType platform_type;
    IntType int_type;
    FloatType float_type;

    KernelContext(KernelType kt, PlatformType pt, IntType it, FloatType ft)
        : kernel_type(kt), platform_type(pt), int_type(it), float_type(ft) {}
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

    // hidden storage for the data pointer:
    // DL 12.05.2025 NOTE: As a workaround for internal D struct in SpTRSV
    void *_val_storage;

    // this is the slot we alias via void**:
    void **val;

    // Default constructor uses our own storage (only for internal structs):
    DenseMatrix()
        : n_rows(0), n_cols(0), _val_storage(nullptr), val(&_val_storage) {}

    // If a user really passes in their own void** (typical case)
    DenseMatrix(int rows, int cols, void **val_ptr = nullptr)
        : n_rows(rows), n_cols(cols), _val_storage(nullptr),
          val(val_ptr ? val_ptr : &_val_storage) {}
};

// TODO
// // Available dense matrix storage formats
// enum DenseMatrixStorageFormat
// {
//     COLWISE,
//     ROWWISE
// };

struct UtilitiesContainer {
    int *lvl_ptr = nullptr;
    int n_levels = 0;

    UtilitiesContainer() = default;

    ~UtilitiesContainer() { delete[] lvl_ptr; }
};

} // namespace SMAX
