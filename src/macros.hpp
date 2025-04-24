#ifndef MACROS_HPP
#define MACROS_HPP

#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CHECK_ERROR(func, label)                   \
    if (func)                                      \
    {                                              \
        std::cerr << "Error in " << label << "\n"; \
        return 1;                                  \
    }

#ifdef _OPENMP

#define GET_THREAD_COUNT(num_threads) \
    int num_threads = omp_get_max_threads();
#define GET_THREAD_ID(tid) \
    int tid = omp_get_thread_num();

#else

#define GET_THREAD_COUNT(num_threads) \
    int num_threads = 1;
#define GET_THREAD_ID(tid) \
    int tid = 0;

#endif

#endif // MACROS_HPP
