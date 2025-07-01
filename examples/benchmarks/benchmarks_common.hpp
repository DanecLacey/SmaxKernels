#pragma once

#include "timing.hpp"

#include <functional>
#include <iostream>
#include <string>

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef CUDA_MODE
#include <cuda_runtime.h>
#endif

#define MIN_BENCH_TIME 1.0
#define MIN_NUM_ITERS 100
#define GF_TO_F 1000000000
#define F_TO_GF 0.000000001

class BenchHarness {
  public:
    const std::string &bench_name;
    std::function<void(bool)> callback;
    int &n_iters;
    float &runtime;
    float min_bench_time;
    int *counter;

    BenchHarness(const std::string &_bench_name,
                 std::function<void(bool)> _callback, int &_n_iters,
                 float &_runtime, float _min_bench_time,
                 int *_counter = nullptr)
        : bench_name(_bench_name), callback(_callback), n_iters(_n_iters),
          runtime(_runtime), min_bench_time(_min_bench_time),
          counter(_counter) {};

    void warmup() {
#ifdef CUDA_MODE
        cudaEvent_t warmup_begin_loop_time, warmup_end_loop_time;
        CUDA_CHECK(cudaEventCreate(&warmup_begin_loop_time));
        CUDA_CHECK(cudaEventCreate(&warmup_end_loop_time));
        CUDA_CHECK(cudaDeviceSynchronize());
#else
        double warmup_begin_loop_time, warmup_end_loop_time;
        warmup_begin_loop_time = warmup_end_loop_time = 0.0;
#endif

        float warmup_runtime = 0.0;
        int warmup_n_iters = n_iters;

        do {
#ifdef CUDA_MODE
            CUDA_CHECK(cudaEventRecord(warmup_begin_loop_time, 0));
#else
            warmup_begin_loop_time = getTimeStamp();
#endif
            for (int k = 0; k < warmup_n_iters; ++k) {
                callback(true);
#ifdef CUDA_MODE
                CUDA_CHECK(cudaDeviceSynchronize());
#endif
            }
            if (counter != nullptr)
                (*counter) += warmup_n_iters;
            warmup_n_iters *= 2;
#ifdef CUDA_MODE
            CUDA_CHECK(cudaEventRecord(warmup_end_loop_time, 0));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventElapsedTime(
                &warmup_runtime, warmup_begin_loop_time, warmup_end_loop_time));
            warmup_runtime /= 1000;
#else
            warmup_end_loop_time = getTimeStamp();
            warmup_runtime = warmup_end_loop_time - warmup_begin_loop_time;
#endif

        } while (warmup_runtime < min_bench_time);
        warmup_n_iters /= 2;
    };

    void bench() {

#ifdef CUDA_MODE
        cudaEvent_t begin_loop_time, end_loop_time;
        CUDA_CHECK(cudaEventCreate(&begin_loop_time));
        CUDA_CHECK(cudaEventCreate(&end_loop_time));
        CUDA_CHECK(cudaDeviceSynchronize());
#else
        double begin_loop_time, end_loop_time;
        begin_loop_time = end_loop_time = 0.0;
#endif

        do {
#ifdef CUDA_MODE
            CUDA_CHECK(cudaEventRecord(begin_loop_time, 0));
#else
            begin_loop_time = getTimeStamp();
#endif
            for (int k = 0; k < n_iters; ++k) {
                callback(false);
#ifdef CUDA_MODE
                CUDA_CHECK(cudaDeviceSynchronize());
#endif
            }
            if (counter != nullptr)
                (*counter) += n_iters;
            n_iters *= 2;

#ifdef CUDA_MODE
            CUDA_CHECK(cudaEventRecord(end_loop_time, 0));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(
                cudaEventElapsedTime(&runtime, begin_loop_time, end_loop_time));
            runtime /= 1000;
#else
            end_loop_time = getTimeStamp();
            runtime = end_loop_time - begin_loop_time;
#endif

        } while (runtime < min_bench_time);
        n_iters /= 2;
    };
};

void init_pin() {
    int n_threads = 1;
    double bogus = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
#endif
#pragma omp parallel for
    for (int i = 0; i < n_threads; ++i) {
        bogus += 1;
    }

    if (bogus < 100) {
        printf(" ");
    }
}

#ifdef USE_LIKWID
#define PARALLEL_LIKWID_MARKER_START(name)                                     \
    do {                                                                       \
        if (!(warmup)) {                                                       \
            _Pragma("omp parallel") { LIKWID_MARKER_START(name); }             \
        }                                                                      \
    } while (0)

#define PARALLEL_LIKWID_MARKER_STOP(name)                                      \
    do {                                                                       \
        if (!(warmup)) {                                                       \
            _Pragma("omp parallel") { LIKWID_MARKER_STOP(name); }              \
        }                                                                      \
    } while (0)

#else

// No‑ops when LIKWID support is disabled:
#define PARALLEL_LIKWID_MARKER_START(name)                                     \
    do {                                                                       \
    } while (0)
#define PARALLEL_LIKWID_MARKER_STOP(name)                                      \
    do {                                                                       \
    } while (0)

#endif

#define RUN_BENCH                                                              \
    BenchHarness *bench_harness =                                              \
        new BenchHarness(bench_name, lambda, n_iter, runtime, MIN_BENCH_TIME); \
    std::cout << "Running bench: " << bench_name << std::endl;                 \
    bench_harness->warmup();                                                   \
    printf("Warmup complete\n");                                               \
    bench_harness->bench();                                                    \
    printf("Bench complete\n")
