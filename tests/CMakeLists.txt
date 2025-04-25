# Set header and library paths
set(INSTALLED_LIB "${CMAKE_INSTALL_PREFIX}/lib")
set(HEADERS "${CMAKE_INSTALL_PREFIX}/include")

# Include and link directories
link_directories(${INSTALLED_LIB})
include_directories(${HEADERS})

# Find all test .cpp files
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test*.cpp")

# Collect target names for top-level access
set(TEST_TARGETS "")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})

    target_compile_options(${test_name} PRIVATE -g -Wall -Wextra -std=c++14)

    # Link to the installed library
    target_link_libraries(${test_name}
        PRIVATE smaxkernels
    )

    # Add rpath so the built binary can find the lib at runtime
    set_target_properties(${test_name} PROPERTIES
        BUILD_RPATH ${INSTALLED_LIB}
    )

    list(APPEND TEST_TARGETS ${test_name})
endforeach()

# Export back to parent scope
set(TEST_TARGETS ${TEST_TARGETS} PARENT_SCOPE)