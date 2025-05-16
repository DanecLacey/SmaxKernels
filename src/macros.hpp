#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

namespace SMAX {

#define CHECK_ERROR(func, label)                                               \
    if (func) {                                                                \
        std::cerr << "Error in " << label << "\n";                             \
        return 1;                                                              \
    }

#ifdef _OPENMP

#define GET_THREAD_COUNT(Type, num_threads)                                    \
    Type num_threads = omp_get_max_threads();
#define GET_THREAD_ID(Type, tid) Type tid = omp_get_thread_num();

#else

#define GET_THREAD_COUNT(Type, num_threads) Type num_threads = 1;
#define GET_THREAD_ID(Type, tid) Type tid = 0;

#endif

#ifdef DEBUG_MODE
#define IF_SMAX_DEBUG(code)                                                    \
    do {                                                                       \
        code;                                                                  \
    } while (0)

#else
#define IF_SMAX_DEBUG(code)                                                    \
    do {                                                                       \
    } while (0)
#endif

#if defined(DEBUG_MODE) && (DEBUG_LEVEL == 1)
#define IF_SMAX_DEBUG_1(...)                                                   \
    do {                                                                       \
        __VA_ARGS__;                                                           \
    } while (0)
#else
#define IF_SMAX_DEBUG_1(...)
#endif

#if defined(DEBUG_MODE) && (DEBUG_LEVEL == 2)
#define IF_SMAX_DEBUG_2(...)                                                   \
    do {                                                                       \
        __VA_ARGS__;                                                           \
    } while (0)
#else
#define IF_SMAX_DEBUG_2(...)
#endif

#if defined(DEBUG_MODE) && (DEBUG_LEVEL == 3)
#define IF_SMAX_DEBUG_3(...)                                                   \
    do {                                                                       \
        __VA_ARGS__;                                                           \
    } while (0)
#else
#define IF_SMAX_DEBUG_3(...)
#endif

#ifdef USE_TIMERS
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
