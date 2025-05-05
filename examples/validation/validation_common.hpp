#ifndef VALIDATION_COMMON_HPP
#define VALIDATION_COMMON_HPP

#include "SmaxKernels/interface.hpp"
#include <mkl.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <math.h>

#define PRINT_WIDTH 18

#ifdef _OPENMP

#define GET_THREAD_COUNT(Type, num_threads)                                    \
    Type num_threads = omp_get_max_threads();
#define GET_THREAD_ID(Type, tid) Type tid = omp_get_thread_num();

#else

#define GET_THREAD_COUNT(Type, num_threads) Type num_threads = 1;
#define GET_THREAD_ID(Type, tid) Type tid = 0;

#endif

double compute_euclid_dist(const int n_rows, const double *y_SMAX,
                           const double *y_MKL) {
    double tmp = 0.0;

#pragma omp parallel for reduction(+ : tmp)
    for (int i = 0; i < n_rows; ++i) {
        tmp += (y_SMAX[i] - y_MKL[i]) * (y_SMAX[i] - y_MKL[i]);
    }

    return std::sqrt(tmp);
}

#endif // VALIDATION_COMMON_HPP