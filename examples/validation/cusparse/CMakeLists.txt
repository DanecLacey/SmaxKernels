# Get all .cpp files in the current directory
file(GLOB CPP_FILES *.cpp)

# Create a list to store target names
set(cusparse_validation_targets)   # this will be passed back up

if(SMAX_CUDA_MODE_EXPORT AND CUDAToolkit_FOUND AND USE_CUSPARSE)
    foreach(src ${CPP_FILES})
        # Get the filename without the directory and extension
        get_filename_component(exec_name ${src} NAME_WE)

        # Add an executable for this source file
        add_executable(${exec_name} ${src})

        # add_compile_definitions(VERBOSITY=${VERBOSITY})
        target_compile_definitions(${exec_name} PRIVATE VERBOSITY=${VERBOSITY})

        # Link against common libraries
        target_link_libraries(${exec_name} PRIVATE SMAX::smaxkernels mmio CUDA::cusparse)

        # Add to the list of targets
        list(APPEND cusparse_validation_targets ${exec_name})
    endforeach()
endif()

# Export the list to parent scope
set(cusparse_validation_targets ${cusparse_validation_targets} PARENT_SCOPE)
