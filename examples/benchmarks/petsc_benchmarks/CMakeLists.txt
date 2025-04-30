add_executable(petsc_spmv petsc_spmv.cpp)

# Link against time stamping
target_link_libraries(petsc_spmv PRIVATE SMAX::smaxkernels timing)

# Link against matrix market reader
target_link_libraries(petsc_spmv PRIVATE SMAX::smaxkernels mmio)

# Link and include PETSc via. PkgConfig
target_link_libraries(petsc_spmv PRIVATE PkgConfig::PETSC)

# Optionally link against LIKWID
if(USE_LIKWID)
    target_link_libraries(petsc_spmv PRIVATE ${LIKWID_LIBRARIES_TO_LINK})
endif()

set(petsc_benchmarks_TARGETS petsc_spmv PARENT_SCOPE)