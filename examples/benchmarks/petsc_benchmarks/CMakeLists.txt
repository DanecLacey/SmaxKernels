add_executable(petsc_spmv petsc_spmv.cpp)

set(petsc_benchmarks_TARGETS petsc_spmv)

# Propagate back to the parent scope
set(petsc_benchmarks_TARGETS ${petsc_benchmarks_TARGETS} PARENT_SCOPE)