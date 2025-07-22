# List of subdirectories to include
set(BENCHMARK_DIRS
    smax_benchmarks
)

if(USE_CUSPARSE)
    list(APPEND BENCHMARK_DIRS cusparse_benchmarks)
endif()
if(USE_EIGEN)
    list(APPEND BENCHMARK_DIRS eigen_benchmarks)
endif()
if(USE_MKL)
    list(APPEND BENCHMARK_DIRS mkl_benchmarks)
endif()

add_library(timing STATIC timing.cpp)
target_include_directories(timing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Clear a global list to collect all app targets
set(benchmark_targets)

foreach(benchmark_dir IN LISTS BENCHMARK_DIRS)
    # Add the benchmark subdirectory
    add_subdirectory(${benchmark_dir})

    # Check if the subdirectory has set its _targets variable
    if(DEFINED ${benchmark_dir}_targets)
        # Append the targets
        list(APPEND benchmark_targets ${${benchmark_dir}_targets})
    else()
        message(WARNING "Benchmark directory '${benchmark_dir}' did not define ${benchmark_dir}_targets")
    endif()
endforeach()

# Export the list of benchmark targets to the parent scope
set(benchmark_targets ${benchmark_targets} PARENT_SCOPE)


