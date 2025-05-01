# Get all .cpp files in the current directory
file(GLOB CPP_FILES *.cpp)

# Create a list to store target names
set(test_TARGETS)

foreach(src ${CPP_FILES})
    # Get the filename without the directory and extension
    get_filename_component(exec_name ${src} NAME_WE)

    # Add an executable for this source file
    add_executable(${exec_name} ${src})

    # Link against common libraries
    target_link_libraries(${exec_name} PRIVATE SMAX::smaxkernels)

    # Add to the list of targets
    list(APPEND test_TARGETS ${exec_name})
endforeach()

# Export the list to parent scope
set(test_TARGETS ${test_TARGETS} PARENT_SCOPE)
