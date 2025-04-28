add_executable(smax_spmv smax_spmv.cpp)

set(smax_benchmarks_TARGETS smax_spmv)

# Propagate back to the parent scope
set(smax_benchmarks_TARGETS ${smax_benchmarks_TARGETS} PARENT_SCOPE)