# Get all .cpp files in the current directory
file(GLOB CPP_FILES *.cpp)

# Create a list to store target names
set(smax_benchmarks_targets)

foreach(src ${CPP_FILES})
    # Get the filename without the directory and extension
    get_filename_component(exec_name ${src} NAME_WE)

    # Add an executable for this source file
    add_executable(${exec_name} ${src})

    # Link against common libraries
    target_link_libraries(${exec_name} PRIVATE SMAX::smaxkernels timing mmio)

    # Optionally link against LIKWID
    if(USE_LIKWID)
        target_link_libraries(${exec_name} PRIVATE ${LIKWID_LIBRARIES_TO_LINK})
    endif()

    # Add to the list of targets
    list(APPEND smax_benchmarks_targets ${exec_name})
endforeach()

# Export the list to parent scope
set(smax_benchmarks_targets ${smax_benchmarks_targets} PARENT_SCOPE)
