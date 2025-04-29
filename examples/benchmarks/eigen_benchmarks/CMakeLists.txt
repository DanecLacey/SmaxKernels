add_executable(eigen_spmv eigen_spmv.cpp)

# Link against time stamping
target_link_libraries(eigen_spmv PRIVATE SMAX::smaxkernels timing)

# Link against matrix market reader
target_link_libraries(eigen_spmv PRIVATE SMAX::smaxkernels mmio)

# Link against the exported SMAX target
# target_link_libraries(eigen_spmv PRIVATE SMAX::smaxkernels)

target_link_libraries(eigen_spmv PRIVATE Eigen3::Eigen)

set(eigen_benchmarks_TARGETS eigen_spmv PARENT_SCOPE)