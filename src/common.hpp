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

struct FlagType {
    bool is_lvl_ptr_collected = false;
    bool is_mat_permuted = false;
};

// Make base class. All child classes will be created for each kernel
// wasteful, but should work for now
struct KernelContext {
    KernelType kernel_type;
    PlatformType platform_type;
    IntType int_type;
    FloatType float_type;
    FlagType flags;
};

// template <KernelType K> struct FlagType {
//     // DL 06.05.2025 TODO: General flags that could be used by all kernels
// };

// template <> struct FlagType<KernelType::SPMV> {
//     // DL 06.05.2025 TODO: Add SpMV specific flags
// };

// template <> struct FlagType<KernelType::SPMM> {
//     // DL 06.05.2025 TODO: Add SpMM specific flags
// };

// template <> struct FlagType<KernelType::SPGEMV> {
//     // DL 06.05.2025 TODO: Add SPGEMV specific flags
// };

// template <> struct FlagType<KernelType::SPGEMM> {
//     // DL 06.05.2025 TODO: Add SPGEMM specific flags
// };

// template <> struct FlagType<KernelType::SPTRSV> {
//     bool is_lvl_ptr_collected = false;
//     bool is_mat_permuted = false;
// };

// template <> struct FlagType<KernelType::SPTRSM> {
//     // DL 06.05.2025 TODO: Add SPTRSM specific flags
// };

// template <KernelType K> struct KernelContext {
//     using IntType = u_int32_t; // Default
//     using FloatType = double;  // Default

//     PlatformType platform_type;
//     IntType int_type;
//     FloatType float_type;
//     FlagType<K> flags;

//     KernelContext(PlatformType platform, IntType int_val, FloatType
//     float_val,
//                   FlagType<K> flag)
//         : platform_type(platform), int_type(int_val), float_type(float_val),
//           flags(flag) {}

//     void display() {
//         std::cout << "KernelType: " << static_cast<int>(kernel_type)
//                   << ", PlatformType: " << static_cast<int>(platform_type)
//                   << ", IntType: " << int_type << ", FloatType: " <<
//                   float_type
//                   << std::endl;
//     }
// };

// template <> struct KernelContext<KernelType::SPTRSV> {
//     FlagType<KernelType::SPTRSV> flags;

//     int *lvl_ptr;
// };

// // Specialization for Spmv
// template <> struct KernelContext<KernelType::SPMV> {
//     using IntType = long long;
//     using FloatType = double;
//     using FlagType = unsigned long long;

//     KernelType kernel_type;
//     PlatformType platform_type;
//     IntType int_type;
//     FloatType float_type;
//     FlagType flags;

//     KernelContext(KernelType kernel, PlatformType platform, IntType int_val,
//                   FloatType float_val, FlagType flag)
//         : kernel_type(kernel), platform_type(platform), int_type(int_val),
//           float_type(float_val), flags(flag) {}

//     void display() {
//         std::cout << "Spmv Kernel - KernelType: "
//                   << static_cast<int>(kernel_type)
//                   << ", PlatformType: " << static_cast<int>(platform_type)
//                   << ", IntType: " << int_type << ", FloatType: " <<
//                   float_type
//                   << ", Flags: " << flags << std::endl;
//     }
// };

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
