# List of subdirectories containing applications
set(APPLICATION_DIRS
    gauss_seidel
)

# Clear a global list to collect all app targets
set(APPLICATION_TARGETS)

foreach(application_dir IN LISTS APPLICATION_DIRS)
    # Add the application subdirectory
    add_subdirectory(${application_dir})

    # Check if the subdirectory has set its _TARGETS variable
    if(DEFINED ${application_dir}_TARGETS)
        # Append the targets
        list(APPEND APPLICATION_TARGETS ${${application_dir}_TARGETS})
    else()
        message(WARNING "Application directory '${application_dir}' did not define ${application_dir}_TARGETS")
    endif()
endforeach()

# Export the list of application targets to the parent scope
set(APPLICATION_TARGETS ${APPLICATION_TARGETS} PARENT_SCOPE)
