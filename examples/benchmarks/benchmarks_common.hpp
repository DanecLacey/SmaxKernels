#ifndef BENCHMARKS_COMMON_HPP
#define BENCHMARKS_COMMON_HPP

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
#define MIN_NUM_ITERS 1000
#define SPMV_FLOPS_PER_NZ 2
#define SPLTSV_FLOPS_PER_NZ 2
#define SPLTSV_FLOPS_PER_ROW 2
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
#ifdef DEBUG_MODE_FINE
                std::cout << "Completed warmup_" << bench_name << " iter " << k
                          << std::endl;
#endif
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
#ifdef DEBUG_MODE_FINE
                std::cout << "Completed " << bench_name << " iter " << k
                          << std::endl;
#endif
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
    int num_threads = 1;
    double bogus = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
#endif
#pragma omp parallel for
    for (int i = 0; i < num_threads; ++i) {
        bogus += 1;
    }

    if (bogus < 100) {
        printf(" ");
    }
}

#ifdef USE_LIKWID
#define IF_USE_LIKWID(cmd) cmd
#else
#define IF_USE_LIKWID(cmd)
#endif

#define RUN_BENCH                                                              \
    BenchHarness *bench_harness =                                              \
        new BenchHarness(bench_name, lambda, n_iter, runtime, MIN_BENCH_TIME); \
    std::cout << "Running bench: " << bench_name << std::endl;                 \
    bench_harness->warmup();                                                   \
    printf("Warmup complete\n");                                               \
    bench_harness->bench();                                                    \
    printf("Bench complete\n");

#define PRINT_SPMV_BENCH                                                       \
    std::cout << "----------------" << std::endl;                              \
    std::cout << "--" << bench_name << " Bench--" << std::endl;                \
    std::cout << cli_args->matrix_file_name << " with " << n_threads           \
              << " thread(s)" << std::endl;                                    \
    std::cout << "Runtime: " << runtime << std::endl;                          \
    std::cout << "Iterations: " << n_iter << std::endl;                        \
                                                                               \
    long flops_per_iter = crs_mat->nnz * SPMV_FLOPS_PER_NZ;                    \
    long iter_per_second = static_cast<long>(n_iter / runtime);                \
                                                                               \
    std::cout << "Performance: " << flops_per_iter * iter_per_second * F_TO_GF \
              << " [GF/s]" << std::endl;                                       \
    std::cout << "----------------" << std::endl;

#define SPMV_CLEANUP                                                           \
    delete cli_args;                                                           \
    delete coo_mat;                                                            \
    delete crs_mat;                                                            \
    delete bench_harness;

#define PRINT_SPTSV_BENCH(bench_name, cli_args, n_threads, runtime, n_iter,    \
                          crs_mat_L)                                           \
    std::cout << "----------------" << std::endl;                              \
    std::cout << "--" << bench_name << " Bench--" << std::endl;                \
    std::cout << (cli_args)->matrix_file_name << " with " << (n_threads)       \
              << " thread(s)" << std::endl;                                    \
    std::cout << "Runtime: " << (runtime) << std::endl;                        \
    std::cout << "Iterations: " << (n_iter) << std::endl;                      \
                                                                               \
    long flops_per_iter = ((crs_mat_L)->nnz * SPLTSV_FLOPS_PER_NZ +            \
                           (crs_mat_L)->n_rows * SPLTSV_FLOPS_PER_ROW);        \
    long iter_per_second = static_cast<long>((n_iter) / (runtime));            \
                                                                               \
    std::cout << "Performance: " << flops_per_iter * iter_per_second * F_TO_GF \
              << " [GF/s]" << std::endl;                                       \
    std::cout << "----------------" << std::endl;

#define SPTSV_CLEANUP                                                          \
    delete cli_args;                                                           \
    delete coo_mat;                                                            \
    delete crs_mat;                                                            \
    delete coo_mat_L;                                                          \
    delete coo_mat_U;                                                          \
    delete crs_mat_L;                                                          \
    delete bench_harness;                                                      \
    delete[] D;                                                                \
    delete[] x;                                                                \
    delete[] b;

#endif // BENCHMARKS_COMMON_HPP