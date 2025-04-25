# List of subdirectories to include
# TODO: Make selection optional
set(BENCHMARK_DIRS
    eigen_benchmarks
    mkl_benchmarks
    petsc_benchmarks
    trilinos_benchmarks
    smax_benchmarks
)

# Iterate over each benchmark directory
foreach(benchmark_dir IN LISTS BENCHMARK_DIRS)
    # Add the subdirectory (it must have its own CMakeLists.txt)
    add_subdirectory(${benchmark_dir})

    # Append the target list for this specific benchmark module
    list(APPEND BENCHMARK_TARGETS ${${benchmark_dir}_TARGETS})
endforeach()

# Propagate the list of benchmark targets to the parent scope
set(BENCHMARK_TARGETS ${BENCHMARK_TARGETS} PARENT_SCOPE)