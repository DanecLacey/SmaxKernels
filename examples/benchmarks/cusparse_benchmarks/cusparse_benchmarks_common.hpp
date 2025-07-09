#pragma once

#define MKL_AGGRESSIVE_N_OPS 10000

#include <cuda_runtime_api.h>
#include <cusparse.h>

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
