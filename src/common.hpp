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
enum class IntType { INT16, INT32, INT64, UINT16, UINT32, UINT64 };

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
    void *col;
    void *row_ptr;
    void *val;

    // Default constructor
    SparseMatrix()
        : n_rows(0), n_cols(0), nnz(0), col(nullptr), row_ptr(nullptr),
          val(nullptr) {}
};

// Workaround for SPGEMM result matrix
struct SparseMatrixRef {
    void *n_rows;
    void *n_cols;
    void *nnz;
    void **col;
    void **row_ptr;
    void **val;

    // Default constructor
    SparseMatrixRef()
        : n_rows(nullptr), n_cols(nullptr), nnz(nullptr), col(nullptr),
          row_ptr(nullptr), val(nullptr) {}
};

// TODO
// // Available sparse matrix storage formats
// enum SparseMatrixStorageFormat
// {
//     CRS,
//     SCS
// };

struct DenseMatrix {
    int n_rows = 0;
    int n_cols = 0;

    // Library-owned storage (optional)
    void *_val_storage = nullptr;

    // Active data pointer (either user-provided or _val_storage)
    void *val = nullptr;

    // Default constructor: no storage allocated
    DenseMatrix() = default;

    // User-managed external memory constructor
    DenseMatrix(int rows, int cols, void *val_ptr = nullptr)
        : n_rows(rows), n_cols(cols), val(val_ptr) {}

    // Destructor: only delete internal storage if it's in use
    ~DenseMatrix() {
        if (val == _val_storage && _val_storage) {
            operator delete(_val_storage);
        }
    }

    // Allocate or reallocate internal storage (library-managed only)
    void allocate_internal(int rows, int cols, size_t elem_size) {
        if (_val_storage) {
            operator delete(_val_storage);
        }
        _val_storage = operator new(rows * cols * elem_size);
        val = _val_storage;

        n_rows = rows;
        n_cols = cols;
    }
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
