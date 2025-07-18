/**
 * @file device_error_handler.hpp
 * @brief Error handling utilities for GPU backends
 *
 * This file provides functions and macros to check and handle errors
 * for CUDA and HIP backends, ensuring proper debugging and error reporting.
 *
 * @author SmaxKernels Team
 * @date 2025
 */

#ifndef DEVICE_ERROR_HANDLER_HPP
#define DEVICE_ERROR_HANDLER_HPP
#pragma once

#include "gpu_backend.hpp"
#include <stdio.h>
#include <stdlib.h>

namespace SMAX
{
    /**
     * @brief Checks the result of a device operation and exits on error
     *
     * @param result The result of the device operation (e.g., error code)
     * @param func The name of the function where the error occurred
     * @param file The name of the file where the error occurred
     * @param line The line number where the error occurred
     *
     * @details If the result is not GPU_BACKEND(Success), this function
     *          prints an error message and exits the program.
     */
    inline void checkDeviceError(GPU_BACKEND(Error_t) result,
                                 char const *const func, const char *const file,
                                 int const line)
    {
        if (result != GPU_BACKEND(Success))
        {
            fprintf(stderr, "%s error at %s:%d code=%d(%s) \"%s\" \n", BACKENDSTR,
                    file, line, static_cast<unsigned int>(result),
                    GPU_BACKEND(GetErrorString)(result), func);
            exit(EXIT_FAILURE);
        }
    }

    /**
     * @brief Checks the last device error and exits on error
     *
     * @param errorMessage A custom error message to display
     * @param file The name of the file where the error occurred
     * @param line The line number where the error occurred
     *
     * @details This function retrieves the last error from the device backend
     *          and exits the program if the error is not GPU_BACKEND(Success).
     */
    inline void checkLastDeviceError(const char *errorMessage, const char *file,
                                     const int line)
    {
        auto err = GPU_BACKEND(GetLastError)();
        if (GPU_BACKEND(Success) != err)
        {
            fprintf(stderr,
                    "%s(%i) : %sgetLastError()  error :"
                    " %s : (%d) %s.\n",
                    file, line, BACKENDSTR, errorMessage, static_cast<int>(err),
                    GPU_BACKEND(GetErrorString)(err));
            exit(EXIT_FAILURE);
        }
    }

    /**
     * @brief Peeks at the last device error and exits on error
     *
     * @param errorMessage A custom error message to display
     * @param file The name of the file where the error occurred
     * @param line The line number where the error occurred
     *
     * @details This function retrieves the last error from the device backend
     *          without clearing it and exits the program if the error is not
     *          GPU_BACKEND(Success).
     */
    inline void peekLastDeviceError(const char *errorMessage, const char *file,
                                    const int line)
    {
        auto err = GPU_BACKEND(PeekAtLastError)();
        if (GPU_BACKEND(Success) != err)
        {
            fprintf(stderr,
                    "%s(%i) : getLast%sError()  error :"
                    " %s : (%d) %s.\n",
                    file, line, BACKENDSTR, errorMessage, static_cast<int>(err),
                    GPU_BACKEND(GetErrorString)(err));
            exit(EXIT_FAILURE);
        }
    }
} // namespace SMAX

/**
 * @brief Macro to check device error
 *
 * @param val The value to check for errors
 *
 * @details This macro calls checkDeviceError with the provided value,
 *          function name, file name, and line number.
 */
#define CHECK_DEVICE_ERR(val) SMAX::checkDeviceError((val), #val, __FILE__, __LINE__);

/**
 * @brief Macro to check the last device error
 *
 * @param ... Custom error message to display
 *
 * @details This macro calls checkLastDeviceError with the provided error
 *          message, file name, and line number.
 */
#define CHECK_DEVICE_LASTERR(...) \
    SMAX::checkLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__);

/**
 * @brief Macro to peek at the last device error
 *
 * @param ... Custom error message to display
 *
 * @details This macro calls peekLastDeviceError with the provided error
 *          message, file name, and line number.
 */
#define PEEK_DEVICE_LASTERR(...) \
    SMAX::peekLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__);

#endif
