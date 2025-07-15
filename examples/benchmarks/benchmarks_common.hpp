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

#define MIN_BENCH_TIME 3.0
#define MIN_NUM_ITERS 100
#define GF_TO_F 1000000000
#define F_TO_GF 0.000000001

#ifdef USE_LIKWID
#define PARALLEL_LIKWID_MARKER_START(name)                                     \
    _Pragma("omp parallel") { LIKWID_MARKER_START(name); }

#define PARALLEL_LIKWID_MARKER_STOP(name)                                      \
    _Pragma("omp parallel") { LIKWID_MARKER_STOP(name); }

#else

// No‑ops when LIKWID support is disabled:
#define PARALLEL_LIKWID_MARKER_START(name)

#define PARALLEL_LIKWID_MARKER_STOP(name)

#endif

class BenchHarness {
  public:
    virtual ~BenchHarness() = default;
    const std::string &bench_name;
    std::function<void()> callback;
    int &n_iters;
    float &runtime;
    float min_bench_time;
    int *counter;

    BenchHarness(const std::string &_bench_name,
                 std::function<void()> _callback, int &_n_iters,
                 float &_runtime, float _min_bench_time,
                 int *_counter = nullptr)
        : bench_name(_bench_name), callback(_callback), n_iters(_n_iters),
          runtime(_runtime), min_bench_time(_min_bench_time),
          counter(_counter) {};

    void warmup() {
        setup_timing();
        float warmup_runtime = 0.f;
        int warmup_iters = n_iters;

        do {
            warmup_record_start();
            for (int i = 0; i < warmup_iters; ++i) {
                callback();
            }
            if (counter)
                *counter += warmup_iters;
            warmup_iters *= 2;
            warmup_record_stop();
            warmup_runtime = compute_elapsed() / time_scale();
        } while (warmup_runtime < min_bench_time);
        warmup_iters /= 2;
    }

    void bench() {
        setup_timing();

        do {
            record_start();
            for (int i = 0; i < n_iters; ++i) {
                callback();
            }
            if (counter)
                *counter += n_iters;
            n_iters *= 2;
            record_stop();
            runtime = compute_elapsed() / time_scale();
        } while (runtime < min_bench_time);

        n_iters /= 2;
    }

  protected:
    virtual void setup_timing() = 0;
    virtual void record_start() = 0;
    virtual void record_stop() = 0;
    virtual void warmup_record_start() = 0;
    virtual void warmup_record_stop() = 0;
    virtual float compute_elapsed() const = 0;
    virtual float time_scale() const = 0;
};

class BenchHarnessCPU : public BenchHarness {
    double begin_loop_time, end_loop_time;

  public:
    using BenchHarness::BenchHarness; // inherit all of the base‐class ctors

  protected:
    void setup_timing() override {}
    void record_start() override {
        PARALLEL_LIKWID_MARKER_START(bench_name.c_str());
        begin_loop_time = getTimeStamp();
    }
    void record_stop() override {
        end_loop_time = getTimeStamp();
        PARALLEL_LIKWID_MARKER_STOP(bench_name.c_str());
    }
    void warmup_record_start() override { begin_loop_time = getTimeStamp(); }
    void warmup_record_stop() override { end_loop_time = getTimeStamp(); }
    float compute_elapsed() const override {
        return end_loop_time - begin_loop_time;
    }
    float time_scale() const override { return 1.0; }
};

class BenchHarnessCUDA : public BenchHarness {
#ifdef CUDA_MODE
    cudaEvent_t start_ev, stop_ev;

  public:
    using BenchHarness::BenchHarness; // inherit all of the base‐class ctors

    // one forwarding ctor that also creates the events exactly once
    template <typename... Args>
    BenchHarnessCUDA(Args &&...args)
        : BenchHarness(std::forward<Args>(args)...) {
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));
    }

    ~BenchHarnessCUDA() override {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

  protected:
    void setup_timing() override { CUDA_CHECK(cudaDeviceSynchronize()); }
    void record_start() override { CUDA_CHECK(cudaEventRecord(start_ev, 0)); }
    void record_stop() override {
        CUDA_CHECK(cudaEventRecord(stop_ev, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
    }
    void warmup_record_start() override {
        CUDA_CHECK(cudaEventRecord(start_ev, 0));
    }
    void warmup_record_stop() override {
        CUDA_CHECK(cudaEventRecord(stop_ev, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
    }
    float compute_elapsed() const override {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        return ms;
    }
    float time_scale() const override { return 1000.0; }
#endif
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

#ifdef CUDA_MODE
using harness_type = BenchHarnessCUDA;
#else
using harness_type = BenchHarnessCPU;
#endif

// helper: OpenMP thread‐counting (no‐op if _OPENMP isn’t defined)
#ifdef _OPENMP
#define SETUP_OMP_THREADS()                                                    \
    _Pragma("omp parallel") { n_threads = omp_get_num_threads(); }
#else
#define SETUP_OMP_THREADS() /* nothing */
#endif

// helper: LIKWID init + register (no‐op if USE_LIKWID isn’t defined)
#ifdef USE_LIKWID
#define INIT_LIKWID_MARKERS(name)                                              \
    do {                                                                       \
        LIKWID_MARKER_INIT;                                                    \
        _Pragma("omp parallel") { LIKWID_MARKER_REGISTER((name).c_str()); }    \
    } while (0)

#define FINALIZE_LIKWID_MARKERS                                                \
    do {                                                                       \
        LIKWID_MARKER_CLOSE;                                                   \
    } while (0)
#else
#define INIT_LIKWID_MARKERS(name)                                              \
    do {                                                                       \
    } while (0)
#define FINALIZE_LIKWID_MARKERS                                                \
    do {                                                                       \
    } while (0)
#endif

#define SETUP_BENCH                                                            \
    float runtime = 0.0f;                                                      \
    int n_iter = MIN_NUM_ITERS;                                                \
    int n_threads = 1;                                                         \
    SETUP_OMP_THREADS();

#define RUN_BENCH                                                              \
    std::unique_ptr<harness_type> bench_harness =                              \
        std::make_unique<harness_type>(bench_name, lambda, n_iter, runtime,    \
                                       MIN_BENCH_TIME);                        \
                                                                               \
    std::cout << "Running bench: " << bench_name << std::endl;                 \
                                                                               \
    bench_harness->warmup();                                                   \
    printf("Warmup complete\n");                                               \
                                                                               \
    bench_harness->bench();                                                    \
    printf("Bench complete\n")
