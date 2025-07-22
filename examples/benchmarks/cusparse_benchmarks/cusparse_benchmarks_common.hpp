#pragma once

#define MKL_AGGRESSIVE_N_OPS 10000

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <memory>

#define CHECK_CUDA(func)                                                       \
    do {                                                                       \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            printf("CUDA API failed at line %d with error: %s (%d)\n",         \
                   __LINE__, cudaGetErrorString(status), status);              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CHECK_CUSPARSE(func)                                                   \
    do {                                                                       \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",     \
                   __LINE__, cusparseGetErrorString(status), status);          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// helper for static_assert in a non‐instantiated branch
template <class> inline constexpr bool always_false = false;

#define DEFINE_CUSPARSE_TYPES(IT, VT)                                          \
    /* pick index‐type enum */                                                 \
    constexpr auto CUSPARSE_INDEX_TYPE = []() {                                \
        if constexpr (std::is_same_v<IT, int>)                                 \
            return CUSPARSE_INDEX_32I;                                         \
        else if constexpr (std::is_same_v<IT, long> ||                         \
                           std::is_same_v<IT, long long>)                      \
            return CUSPARSE_INDEX_64I;                                         \
    }();                                                                       \
    /* pick value‐type enum */                                                 \
    constexpr auto CUSPARSE_FLOAT_TYPE = []() {                                \
        if constexpr (std::is_same_v<VT, float>)                               \
            return CUDA_R_32F;                                                 \
        else if constexpr (std::is_same_v<VT, double>)                         \
            return CUDA_R_64F;                                                 \
    }();
