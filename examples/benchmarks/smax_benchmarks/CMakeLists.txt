add_executable(smax_spmv smax_spmv.cpp)

# Link against time stamping
target_link_libraries(smax_spmv PRIVATE SMAX::smaxkernels timing)

# Link against matrix market reader
target_link_libraries(smax_spmv PRIVATE SMAX::smaxkernels mmio)

# Link against the exported SMAX target
target_link_libraries(smax_spmv PRIVATE SMAX::smaxkernels)

# Optionally link against LIKWID
if(USE_LIKWID)
    target_link_libraries(smax_spmv PRIVATE ${LIKWID_LIBRARIES_TO_LINK})
endif()

set(smax_benchmarks_TARGETS smax_spmv PARENT_SCOPE)