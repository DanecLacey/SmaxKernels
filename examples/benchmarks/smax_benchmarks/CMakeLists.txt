add_executable(smax_spmv smax_spmv.cpp)

# Link against time stamping
target_link_libraries(smax_spmv PRIVATE SMAX::smaxkernels timing)

# Link against matrix market reader
target_link_libraries(smax_spmv PRIVATE SMAX::smaxkernels mmio)

# Link against the exported SMAX target
target_link_libraries(smax_spmv PRIVATE SMAX::smaxkernels)

set(smax_benchmarks_TARGETS smax_spmv PARENT_SCOPE)