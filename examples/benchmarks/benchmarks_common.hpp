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

#define MIN_BENCH_TIME 1.0
#define MIN_NUM_ITERS 100
#define GF_TO_F 1000000000
#define F_TO_GF 0.000000001

class BenchHarness {
  public:
    const std::string &bench_name;
    std::function<void(bool)> callback;
    int &n_iters;
    double &runtime;
    double min_bench_time;
    int *counter;

    BenchHarness(const std::string &_bench_name,
                 std::function<void(bool)> _callback, int &_n_iters,
                 double &_runtime, double _min_bench_time,
                 int *_counter = nullptr)
        : bench_name(_bench_name), callback(_callback), n_iters(_n_iters),
          runtime(_runtime), min_bench_time(_min_bench_time),
          counter(_counter) {};

    void warmup() {
        double warmup_begin_loop_time, warmup_end_loop_time;
        warmup_begin_loop_time = warmup_end_loop_time = 0.0;

        int warmup_n_iters = n_iters;
        double warmup_runtime = 0.0;

        do {
            warmup_begin_loop_time = getTimeStamp();
            for (int k = 0; k < warmup_n_iters; ++k) {
                callback(true);
            }
            if (counter != nullptr)
                (*counter) += warmup_n_iters;
            warmup_n_iters *= 2;
            warmup_end_loop_time = getTimeStamp();
            warmup_runtime = warmup_end_loop_time - warmup_begin_loop_time;
        } while (warmup_runtime < min_bench_time);
        warmup_n_iters /= 2;
    };

    void bench() {
        double begin_loop_time, end_loop_time;
        begin_loop_time = end_loop_time = 0.0;

        do {
            begin_loop_time = getTimeStamp();
            for (int k = 0; k < n_iters; ++k) {
                callback(false);
            }
            if (counter != nullptr)
                (*counter) += n_iters;
            n_iters *= 2;
            end_loop_time = getTimeStamp();
            runtime = end_loop_time - begin_loop_time;
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
