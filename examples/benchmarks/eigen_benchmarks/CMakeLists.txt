add_executable(eigen_spmv eigen_spmv.cpp)

# Link against time stamping
target_link_libraries(eigen_spmv PRIVATE SMAX::smaxkernels timing)

# Link against matrix market reader
target_link_libraries(eigen_spmv PRIVATE SMAX::smaxkernels mmio)

# Link against public Eigen CMake package 
target_link_libraries(eigen_spmv PRIVATE Eigen3::Eigen)

# Optionally link against LIKWID
if(USE_LIKWID)
    target_link_libraries(eigen_spmv PRIVATE ${LIKWID_LIBRARIES_TO_LINK})
endif()

set(eigen_benchmarks_TARGETS eigen_spmv PARENT_SCOPE)