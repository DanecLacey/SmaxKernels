# Get all .cpp files in the current directory
file(GLOB CPP_FILES *.cpp)

# Create a list to store target names
set(validation_targets)

if(MKL_USE_ILP64)
    message(STATUS "Validation targets will link with MKL ILP64")
else()
    message(STATUS "Validation targets will link with MKL LP64")
endif()

foreach(src ${CPP_FILES})
    # Get the filename without the directory and extension
    get_filename_component(exec_name ${src} NAME_WE)

    # Add an executable for this source file
    add_executable(${exec_name} ${src})

    # add_compile_definitions(VERBOSITY=${VERBOSITY})
    target_compile_definitions(${exec_name} PRIVATE VERBOSITY=${VERBOSITY})

    # Link against common libraries
    target_link_libraries(${exec_name} PRIVATE SMAX::smaxkernels mmio MKL::MKL)

    # Just to be safe
    if(MKL_USE_ILP64)
        target_compile_definitions(${exec_name} PRIVATE USE_MKL_ILP64)
    else()
        target_compile_definitions(${exec_name} PRIVATE USE_MKL_LP64)
    endif()

    # Add to the list of targets
    list(APPEND validation_targets ${exec_name})
endforeach()

# Export the list to parent scope
set(validation_targets ${validation_targets} PARENT_SCOPE)
