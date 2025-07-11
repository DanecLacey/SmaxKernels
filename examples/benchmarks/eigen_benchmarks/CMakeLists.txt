# Get all .cpp files in the current directory
file(GLOB CPP_FILES *.cpp)


# Create a list to store target names
set(eigen_benchmarks_targets)

foreach(src ${CPP_FILES})
    # Get the filename without the directory and extension
    get_filename_component(exec_name ${src} NAME_WE)

    # Add an executable for this source file
    add_executable(${exec_name} ${src})

    # Link against common libraries
    target_link_libraries(${exec_name} PRIVATE SMAX::smaxkernels timing mmio)

    # Link against Eigen CMake export package
    target_link_libraries(${exec_name} PRIVATE Eigen3::Eigen)

    # Optionally link against LIKWID
    if(USE_LIKWID)
        target_link_libraries(${exec_name} PRIVATE ${LIKWID_LIBRARIES_TO_LINK})
    endif()

    # Add to the list of targets
    list(APPEND eigen_benchmarks_targets ${exec_name})

    target_compile_options(${exec_name} PRIVATE -mavx512f)
    if(CPU_COMPILER_ID STREQUAL "GNU" OR CPU_COMPILER_ID STREQUAL "INTEL")
        target_compile_options(${exec_name} PRIVATE -mfma -mavx -msse)
        string(REPLACE "-march=native" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif()
    target_compile_definitions(${exec_name} PRIVATE EIGEN_VECTORIZE_AVX512)

endforeach()

# Export the list to parent scope
set(eigen_benchmarks_targets ${eigen_benchmarks_targets} PARENT_SCOPE)