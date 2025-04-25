# List of subdirectories to include
# TODO: Make selection optional
set(APPLICATION_DIRS
    test_app
)

# Iterate over each application directory
foreach(application_dir IN LISTS APPLICATION_DIRS)
    # Add the subdirectory (it must have its own CMakeLists.txt)
    add_subdirectory(${application_dir})

    # Append the target list for this specific application module
    list(APPEND APPLICATION_TARGETS ${${application_dir}_TARGETS})
endforeach()

# Propagate the list of application targets to the parent scope
set(APPLICATION_TARGETS ${APPLICATION_TARGETS} PARENT_SCOPE)