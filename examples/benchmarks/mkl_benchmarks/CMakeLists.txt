add_executable(mkl_spmv mkl_spmv.cpp)

# Link against time stamping
target_link_libraries(mkl_spmv PRIVATE SMAX::smaxkernels timing)

# Link against matrix market reader
target_link_libraries(mkl_spmv PRIVATE SMAX::smaxkernels mmio)

# Link against public MKL CMake package
target_link_libraries(mkl_spmv PUBLIC MKL::MKL)

# Optionally link against LIKWID
if(USE_LIKWID)
    target_link_libraries(mkl_spmv PRIVATE ${LIKWID_LIBRARIES_TO_LINK})
endif()

set(mkl_benchmarks_TARGETS mkl_spmv PARENT_SCOPE)