#pragma once

#include "smax_config.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef SMAX_USE_LIKWID
#include <likwid-marker.h>
#endif

namespace SMAX {

#define SMAX_RESTRICT __restrict__

#define SMAX_CHECK_ERROR(func, label)                                          \
    if (func) {                                                                \
        std::cerr << "Error in " << label << "\n";                             \
        return 1;                                                              \
    }

#ifdef _OPENMP

#define SMAX_GET_THREAD_COUNT(Type, n_threads)                                 \
    Type n_threads = omp_get_max_threads();
#define SMAX_GET_THREAD_ID(Type, tid) Type tid = omp_get_thread_num();

#else

#define SMAX_GET_THREAD_COUNT(Type, n_threads) Type n_threads = 1;
#define SMAX_GET_THREAD_ID(Type, tid) Type tid = 0;

#endif

#if SMAX_DEBUG_MODE
#define IF_SMAX_DEBUG(code)                                                    \
    do {                                                                       \
        code;                                                                  \
    } while (0)

#else
#define IF_SMAX_DEBUG(code)                                                    \
    do {                                                                       \
    } while (0)
#endif

#if SMAX_DEBUG_MODE && (SMAX_DEBUG_LEVEL == 1)
#define IF_SMAX_DEBUG_1(...)                                                   \
    do {                                                                       \
        __VA_ARGS__;                                                           \
    } while (0)
#else
#define IF_SMAX_DEBUG_1(...)
#endif

#if SMAX_DEBUG_MODE && (SMAX_DEBUG_LEVEL == 2)
#define IF_SMAX_DEBUG_2(...)                                                   \
    do {                                                                       \
        __VA_ARGS__;                                                           \
    } while (0)
#else
#define IF_SMAX_DEBUG_2(...)
#endif

#if SMAX_DEBUG_MODE && (SMAX_DEBUG_LEVEL == 3)
#define IF_SMAX_DEBUG_3(...)                                                   \
    do {                                                                       \
        __VA_ARGS__;                                                           \
    } while (0)
#else
#define IF_SMAX_DEBUG_3(...)
#endif

#if SMAX_USE_TIMERS
#define IF_SMAX_TIME(code)                                                     \
    do {                                                                       \
        code;                                                                  \
    } while (0)
#else
#define IF_SMAX_TIME(code)                                                     \
    do {                                                                       \
    } while (0)
#endif

} // namespace SMAX
