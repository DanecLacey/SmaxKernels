add_executable(trilinos_spmv trilinos_spmv.cpp)

set(trilinos_benchmarks_TARGETS trilinos_spmv)

# Propagate back to the parent scope
set(trilinos_benchmarks_TARGETS ${trilinos_benchmarks_TARGETS} PARENT_SCOPE)