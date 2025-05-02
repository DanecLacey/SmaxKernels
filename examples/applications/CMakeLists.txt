# List of subdirectories containing applications
set(APPLICATION_DIRS
    gauss_seidel
)

# Clear a global list to collect all app targets
set(application_targets)

foreach(application_dir IN LISTS APPLICATION_DIRS)
    # Add the application subdirectory
    add_subdirectory(${application_dir})

    # Check if the subdirectory has set its _targets variable
    if(DEFINED ${application_dir}_targets)
        # Append the targets
        list(APPEND application_targets ${${application_dir}_targets})
    else()
        message(WARNING "Application directory '${application_dir}' did not define ${application_dir}_targets")
    endif()
endforeach()

# Export the list of application targets to the parent scope
set(application_targets ${application_targets} PARENT_SCOPE)
