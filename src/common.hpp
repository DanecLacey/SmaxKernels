#pragma once

#include "error_handler.hpp"
#include "macros.hpp"
#include "memory_utils.hpp"
#include "stopwatch.hpp"

#include <variant>

namespace SMAX {

using Variant = std::variant<int, void *, void **>;

// Available kernels
enum class KernelType { SPMV, SPMM, SPGEMV, SPGEMM, SPTRSV, SPTRSM };

// Available platforms
enum class PlatformType { CPU, CUDA };

// Available integer types
enum class IntType { INT16, INT32, INT64, UINT16, UINT32, UINT64 };

// Available floating point types
enum class FloatType { FLOAT32, FLOAT64 };

// Internal flags
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

struct CRSMatrix {
    int n_rows = 0;
    int n_cols = 0;
    int nnz = 0;
    void *col = nullptr;
    void *row_ptr = nullptr;
    void *val = nullptr;

    CRSMatrix() = default;

    CRSMatrix(int _n_rows, int _n_cols, int _nnz, void *_col, void *_row_ptr,
              void *_val)
        : n_rows(_n_rows), n_cols(_n_cols), nnz(_nnz), col(_col),
          row_ptr(_row_ptr), val(_val) {}
};

struct SCSMatrix {
    int C = 0;
    int sigma = 0;
    int n_rows = 0;
    int n_rows_padded = 0;
    int n_cols = 0;
    int n_chunks = 0;
    int n_elements = 0;
    int nnz = 0;
    void *chunk_ptr = nullptr;
    void *chunk_lengths = nullptr;
    void *col = nullptr;
    void *val = nullptr;
    void *perm = nullptr;
    void *inv_perm = nullptr;

    SCSMatrix() = default;

    SCSMatrix(int _C, int _sigma, int _n_rows, int _n_rows_padded, int _n_cols,
              int _n_chunks, int _n_elements, int _nnz, void *_chunk_ptr,
              void *_chunk_lengths, void *_col, void *_val, void *_perm,
              void *_inv_perm)
        : C(_C), sigma(_sigma), n_rows(_n_rows), n_rows_padded(_n_rows_padded),
          n_cols(_n_cols), n_chunks(_n_chunks), n_elements(_n_elements),
          nnz(_nnz), chunk_ptr(_chunk_ptr), chunk_lengths(_chunk_lengths),
          col(_col), val(_val), perm(_perm), inv_perm(_inv_perm) {}
};

struct SparseMatrix {
    // Format-specific representations
    std::unique_ptr<CRSMatrix> crs;
    std::unique_ptr<SCSMatrix> scs;

    SparseMatrix() = default;

    ~SparseMatrix() = default;
};

struct CRSMatrixRef {
    void *n_rows = nullptr;
    void *n_cols = nullptr;
    void *nnz = nullptr;
    void **col = nullptr;
    void **row_ptr = nullptr;
    void **val = nullptr;

    CRSMatrixRef() = default;

    CRSMatrixRef(void *_n_rows, void *_n_cols, void *_nnz, void **_col,
                 void **_row_ptr, void **_val)
        : n_rows(_n_rows), n_cols(_n_cols), nnz(_nnz), col(_col),
          row_ptr(_row_ptr), val(_val) {}
};

// Workaround for SPGEMM result matrix
struct SparseMatrixRef {
    // Format-specific representations
    std::unique_ptr<CRSMatrixRef> crs;

    SparseMatrixRef() = default;

    ~SparseMatrixRef() = default;
};

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

struct UtilitiesContainer {
    int *lvl_ptr = nullptr;
    int n_levels = 0;

    UtilitiesContainer() = default;

    ~UtilitiesContainer() { delete[] lvl_ptr; }
};

} // namespace SMAX
