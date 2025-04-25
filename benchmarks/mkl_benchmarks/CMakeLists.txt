add_executable(mkl_spmv mkl_spmv.cpp)

set(mkl_benchmarks_TARGETS mkl_spmv)

# Propagate back to the parent scope
set(mkl_benchmarks_TARGETS ${mkl_benchmarks_TARGETS} PARENT_SCOPE)