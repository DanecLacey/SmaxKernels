#pragma once

#include "SmaxKernels/interface.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

#include "matrix_structs.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#define PRINT_WIDTH 18
#define CACHE_LINE_ALIGNMENT 64

#ifdef _OPENMP
#define GET_THREAD_COUNT ULL n_threads = omp_get_max_threads();
#define GET_THREAD_ID ULL tid = omp_get_thread_num();
#else
#define GET_THREAD_COUNT ULL n_threads = 1;
#define GET_THREAD_ID ULL tid = 0;
#endif

#define CHECK_MKL_STATUS(status, message)                                      \
    if ((status) != SPARSE_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "ERROR: %s\n", (message));                             \
        exit(EXIT_FAILURE);                                                    \
    }

#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#ifdef DEBUG_MODE
#define DIFF_STATUS_MACRO(relative_diff, working_file)                         \
    do {                                                                       \
        if ((std::abs(relative_diff) > 0.01) || std::isinf(relative_diff)) {   \
            working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";    \
        } else if (std::abs(relative_diff) > 0.0001) {                         \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
        }                                                                      \
    } while (0)
#else
#define DIFF_STATUS_MACRO(relative_diff, working_file)                         \
    do {                                                                       \
        if (std::abs(relative_diff) > 0.0001)                                  \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
    } while (0)
#endif

#define UPDATE_MAX_DIFFS(i, y_MKL, y_SMAX, relative_diff, absolute_diff)       \
    do {                                                                       \
        if ((relative_diff) > max_relative_diff) {                             \
            max_relative_diff = (relative_diff);                               \
            max_relative_diff_elem_SMAX = (y_SMAX)[(i)];                       \
            max_relative_diff_elem_MKL = (y_MKL)[(i)];                         \
        }                                                                      \
        if ((absolute_diff) > max_absolute_diff) {                             \
            max_absolute_diff = (absolute_diff);                               \
            max_absolute_diff_elem_SMAX = (y_SMAX)[(i)];                       \
            max_absolute_diff_elem_MKL = (y_MKL)[(i)];                         \
        }                                                                      \
    } while (0)

#ifdef DEBUG_MODE
#define CHECK_MAX_DIFFS_AND_PRINT_ERROR_WARNING(                               \
    max_relative_diff, max_absolute_diff, working_file)                        \
    do {                                                                       \
        if ((((std::abs(max_relative_diff) > 0.01) ||                          \
              std::isnan(max_relative_diff) ||                                 \
              std::isinf(max_relative_diff)) ||                                \
             (std::isnan(max_absolute_diff) ||                                 \
              std::isinf(max_absolute_diff)))) {                               \
            working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";    \
        } else if (std::abs(max_relative_diff) > 0.0001) {                     \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
        }                                                                      \
    } while (0)
#else
#define CHECK_MAX_DIFFS_AND_PRINT_ERROR_WARNING(                               \
    max_relative_diff, max_absolute_diff, working_file)                        \
    do {                                                                       \
        if (std::abs(max_relative_diff) > 0.0001)                              \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
    } while (0)
#endif

void *aligned_malloc(size_t bytesize) {
    int errorCode;
    void *ptr;

    errorCode = posix_memalign(&ptr, CACHE_LINE_ALIGNMENT, bytesize);

    if (errorCode) {
        if (errorCode == EINVAL) {
            fprintf(stderr,
                    "Error: Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM) {
            fprintf(stderr,
                    "Error: Insufficient memory to fulfill the request\n");
            exit(EXIT_FAILURE);
        }
    }

    if (ptr == NULL) {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

// overload new and delete for alignement
void *operator new(size_t bytesize) {
    // printf("Overloading new operator with size: %lu\n", bytesize);
    int errorCode;
    void *ptr;
    errorCode = posix_memalign(&ptr, CACHE_LINE_ALIGNMENT, bytesize);

    if (errorCode) {
        if (errorCode == EINVAL) {
            fprintf(stderr,
                    "Error: Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM) {
            fprintf(stderr, "Error: Insufficient memory to fulfill the request "
                            "for space\n");
            exit(EXIT_FAILURE);
        }
    }

    if (ptr == NULL) {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

void operator delete(void *p) {
    // printf("Overloading delete operator\n");
    free(p);
}

/*
Only type supported by this helper function at this time:
- int, long int, long long int, unsigned long long int
- float, double
*/
// TODO: Clean the mess up!
template <typename IT, typename VT>
void register_kernel(SMAX::Interface *smax, std::string kernel_name,
                     SMAX::KernelType KernelType,
                     SMAX::PlatformType PlatformType) {
    if constexpr (std::is_same_v<IT, int>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT32,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT32,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else if constexpr (std::is_same_v<IT, long>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT64,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT64,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else if constexpr (std::is_same_v<IT, unsigned int>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::UINT32,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::UINT32,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else if constexpr (std::is_same_v<IT, long long>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT64,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT64,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else if constexpr (std::is_same_v<IT, unsigned long long>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::UINT64,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::UINT64,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else {
        std::cout << "IT not recognized" << std::endl;
    }
};

double compute_euclid_dist(const ULL n_rows, const double *y_SMAX,
                           const double *y_MKL) {
    double tmp = 0.0;

#pragma omp parallel for reduction(+ : tmp)
    for (ULL i = 0; i < n_rows; ++i) {
        tmp += (y_SMAX[i] - y_MKL[i]) * (y_SMAX[i] - y_MKL[i]);
    }

    return std::sqrt(tmp);
}

// DL 4.5.25 TODO: Probably a better idea to leave args_ private and add an
// accessor
class CliParser {
  public:
    struct CliArgs {
        std::string matrix_file_name;
        virtual ~CliArgs() = default;
    };

  protected:
    CliArgs *args_;

  public:
    CliParser() : args_(nullptr) {}
    virtual ~CliParser() { delete args_; }

    virtual CliArgs *parse(int argc, char *argv[]) {
        // supress warnings
        (void)argc;
        (void)argv;

        delete args_;
        args_ = new CliArgs();
        return args_;
    }

    CliArgs *args() const { return args_; }
};

template <typename IT, typename VT>
void extract_D_L_U(const CRSMatrix<IT, VT> &A, CRSMatrix<IT, VT> &D_plus_L,
                   CRSMatrix<IT, VT> &U) {
    // Count nnz
    for (ULL i = 0; i < A.n_rows; ++i) {
        IT row_start = A.row_ptr[i];
        IT row_end = A.row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (IT idx = row_start; idx < row_end; ++idx) {
            IT col = A.col[idx];

            if (static_cast<ULL>(col) <= i) {
                ++D_plus_L.nnz;
            } else {
                ++U.nnz;
            }
        }
    }

    // Allocate heap space and assign known metadata
    D_plus_L.val = new VT[D_plus_L.nnz];
    D_plus_L.col = new IT[D_plus_L.nnz];
    D_plus_L.row_ptr = new IT[A.n_rows + 1];
    D_plus_L.row_ptr[0] = 0;
    D_plus_L.n_rows = A.n_rows;
    D_plus_L.n_cols = A.n_cols;

    U.val = new VT[U.nnz];
    U.col = new IT[U.nnz];
    U.row_ptr = new IT[A.n_rows + 1];
    U.row_ptr[0] = 0;
    U.n_rows = A.n_rows;
    U.n_cols = A.n_cols;

    // Assign nonzeros
    ULL D_plus_L_count = 0;
    ULL U_count = 0;
    for (ULL i = 0; i < A.n_rows; ++i) {
        IT row_start = A.row_ptr[i];
        IT row_end = A.row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (IT idx = row_start; idx < row_end; ++idx) {
            IT col = A.col[idx];
            VT val = A.val[idx];

            if (static_cast<ULL>(col) <= i) {
                // Diagonal or lower triangular part (D + L)
                D_plus_L.val[D_plus_L_count] = val;
                D_plus_L.col[D_plus_L_count++] = col;
            } else {
                // Strictly upper triangular part (U)
                U.val[U_count] = val;
                U.col[U_count++] = col;
            }
        }

        // Update row pointers
        D_plus_L.row_ptr[i + 1] = D_plus_L_count;
        U.row_ptr[i + 1] = U_count;
    }
}

template <typename VT> void print_vector(VT *vec, ULL n_rows) {
    printf("Vector: [");
    for (ULL i = 0; i < n_rows; ++i) {
        std::cout << vec[i] << ", ";
    }
    printf("]\n\n");
}

template <typename VT> void print_exact_vector(VT *vec, ULL n_rows) {
#include <iomanip>
#include <iostream>
#include <limits>

    printf("Vector: [");
    for (ULL i = 0; i < n_rows; ++i) {
        std::cout << std::fixed
                  << std::setprecision(
                         std::numeric_limits<double>::max_digits10)
                  << vec[i] << std::endl;
    }
    printf("]\n\n");
}

// template <typename IT, typename VT>
// void print_matrix(ULL n_rows, ULL n_cols, ULL nnz, IT *col, IT *row_ptr,
//                   VT *val, bool symbolic = false) {

//     std::cout << "n_rows = " << n_rows << std::endl;
//     std::cout << "n_cols = " << n_cols << std::endl;
//     std::cout << "nnz = " << nnz << std::endl;

//     printf("col = [");
//     for (ULL i = 0; i < nnz; ++i) {
//         std::cout << col[i] << ", ";
//     }
//     printf("]\n");

//     printf("row_ptr = [");
//     for (ULL i = 0; i <= n_rows; ++i) {
//         std::cout << row_ptr[i] << ", ";
//     }
//     printf("]\n");

//     if (!symbolic) {
//         printf("val = [");
//         for (ULL i = 0; i < nnz; ++i) {
//             std::cout << val[i] << ", ";
//         }
//         printf("]\n");
//     }
//     printf("\n");
// }

// Just for unit tests. In practice, we leave nonzeros in a row unsorted
template <typename IT, typename ST>
void sort_csr_rows_by_col(IT *row_ptr, IT *col, ST n_rows, ST nnz) {
    for (ST i = 0; i < n_rows; ++i) {
        IT *row_start = col + row_ptr[i];
        IT *row_end = col + row_ptr[i + 1];

        std::sort(row_start, row_end);
    }
};
