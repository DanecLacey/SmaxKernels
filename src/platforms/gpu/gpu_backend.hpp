/**
 * @file device_backend.hpp
 * @brief Device backend abstraction layer for CUDA and HIP platforms
 * 
 * This file provides a unified interface for GPU computing backends,
 * automatically detecting and configuring the appropriate runtime
 * (CUDA or HIP) based on the compiler being used.
 * 
 * @author SmaxKernels Team
 * @date 2025
 */

#ifndef GPU_BACKEND_HPP
#define GPU_BACKEND_HPP
#pragma once

/**
 * @brief Backend detection and configuration
 * 
 * Automatically detects the GPU backend based on compiler definitions:
 * - __HIPCC__: AMD HIP backend
 * - __CUDACC__: NVIDIA CUDA backend
 */
#if SMAX_HIP_MODE
/** @brief Backend identifier for HIP */
#define BACKEND hip
/** @brief Human-readable backend string for HIP */
#define BACKENDSTR "HIP"
#include <hip/hip_runtime.h>
#elif SMAX_CUDA_MODE
/** @brief Backend identifier for CUDA */
#define BACKEND cuda
/** @brief Human-readable backend string for CUDA */
#define BACKENDSTR "CUDA"
#include <cuda_runtime.h>

#else
#error "Please define either SMAX_CUDA_MODE or SMAX_HIP_MODE"
#endif

/**
 * @defgroup MacroUtilities Macro Utilities
 * @brief Utility macros for token pasting and backend function naming
 * @{
 */

/**
 * @brief Low-level token pasting macro
 * @param a First token to paste
 * @param b Second token to paste
 * @return Concatenated token a##b
 * 
 * This is an internal helper macro used by APPEND_NAME.
 * Direct usage is not recommended.
 */
#define STRINGIFY_AND_APPEND(a, b) a##b

/**
 * @brief Macro expansion and token pasting utility
 * @param a First token (will be expanded before pasting)
 * @param b Second token (will be expanded before pasting)
 * @return Concatenated token after macro expansion
 * 
 * This macro ensures that both arguments are fully expanded
 * before concatenation, which is necessary for proper macro
 * substitution with the BACKEND macro.
 */
#define APPEND_NAME(a, b) STRINGIFY_AND_APPEND(a, b)

/**
 * @brief Creates backend-specific function names
 * @param name Function or symbol name to be prefixed with backend
 * @return Backend-specific name (e.g., hipname or cudaname)
 * 
 * This macro automatically prefixes the given name with the
 * appropriate backend identifier (hip or cuda), allowing for
 * unified API calls that resolve to backend-specific functions.
 * 
 * Example usage:
 * GPU_BACKEND(Malloc)(ptr, size);
 * Resolves to 
 * hipMalloc(ptr, size) or cudaMalloc(ptr, size)
 */
#define GPU_BACKEND(name) APPEND_NAME(BACKEND, name)

/** @} */ // end of MacroUtilities group

#endif