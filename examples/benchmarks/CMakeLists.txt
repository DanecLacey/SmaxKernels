# List of subdirectories to include
set(BENCHMARK_DIRS)

if(USE_SMAX)
    list(APPEND BENCHMARK_DIRS smax_benchmarks)
endif()
if(USE_EIGEN)
    list(APPEND BENCHMARK_DIRS eigen_benchmarks)
endif()
if(USE_MKL)
    list(APPEND BENCHMARK_DIRS mkl_benchmarks)
endif()
if(USE_PETSC)
    list(APPEND BENCHMARK_DIRS petsc_benchmarks)
endif()

add_library(timing STATIC timing.cpp)
target_include_directories(timing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Clear a global list to collect all app targets
set(BENCHMARK_TARGETS)

foreach(benchmark_dir IN LISTS BENCHMARK_DIRS)
    # Add the benchmark subdirectory
    add_subdirectory(${benchmark_dir})

    # Check if the subdirectory has set its _TARGETS variable
    if(DEFINED ${benchmark_dir}_TARGETS)
        # Append the targets
        list(APPEND BENCHMARK_TARGETS ${${benchmark_dir}_TARGETS})
    else()
        message(WARNING "Benchmark directory '${benchmark_dir}' did not define ${benchmark_dir}_TARGETS")
    endif()
endforeach()

# Export the list of benchmark targets to the parent scope
set(BENCHMARK_TARGETS ${BENCHMARK_TARGETS} PARENT_SCOPE)


