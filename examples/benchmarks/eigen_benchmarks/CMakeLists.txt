add_executable(eigen_spmv eigen_spmv.cpp)

set(eigen_benchmarks_TARGETS eigen_spmv)

# Propagate back to the parent scope
set(eigen_benchmarks_TARGETS ${eigen_benchmarks_TARGETS} PARENT_SCOPE)