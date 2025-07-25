# Get all .cpp files in the current directory
file(GLOB ALL_CPP_FILES *.cpp)

# Create filtered list
set(CPP_FILES "")

if(NOT (SMAX_CUDA_MODE_EXPORT AND CUDAToolkit_FOUND))
    # Skip everything containing keyword "cuda"
    set(SKIP_ME "cuda")
    foreach(file ${ALL_CPP_FILES})
        get_filename_component(FILENAME "${file}" NAME)
        string(FIND "${FILENAME}" "${SKIP_ME}" FOUND)
        if(FOUND EQUAL -1)
            list(APPEND CPP_FILES "${file}")
        else()
            message(STATUS "Skipping ${file} (filename contains '${SKIP_ME}')")
        endif()
    endforeach()
else()
    set(CPP_FILES ${ALL_CPP_FILES})
endif()

# Create a list to store target names
set(demo_targets)

foreach(src ${CPP_FILES})
    # Get the filename without the directory and extension
    get_filename_component(exec_name ${src} NAME_WE)

    # Add an executable for this source file
    add_executable(${exec_name} ${src})

    # Link against common libraries
    target_link_libraries(${exec_name} PRIVATE SMAX::smaxkernels)

    # Add to the list of targets
    list(APPEND demo_targets ${exec_name})
endforeach()

# Export the list to parent scope
set(demo_targets ${demo_targets} PARENT_SCOPE)
