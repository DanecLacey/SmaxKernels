cmake_minimum_required(VERSION 3.23.1)

set(PROJECT_NAME "SmaxKernels")
set(LIBRARY_NAME "smaxkernels")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

# Source and header files for the library
set(SOURCES
    src/interface.cpp
    src/error_handler.cpp
)

set(HEADERS
    src/interface.hpp
    src/common.hpp
    src/macros.hpp
    src/memory_utils.hpp
    src/stopwatch.hpp
    src/kernel.hpp
    src/error_handler.hpp
)

set(HEADER_DIRS
    src/kernels
)

# Define the shared library
add_library(${LIBRARY_NAME} SHARED ${SOURCES})

include(GNUInstallDirs)

# Expose headers during build and installation
target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

# Options for building the library
option(USE_TIMERS "Enable Timers support" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_MPI "Enable MPI support" OFF)

# Install the library
install(TARGETS ${LIBRARY_NAME}
    EXPORT SmaxKernelsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_RUNDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Copy headers over to the installation
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})
install(DIRECTORY ${HEADER_DIRS} DESTINATION include/${PROJECT_NAME})

# Install export for targets
install(EXPORT SmaxKernelsTargets
    FILE SmaxKernelsTargets.cmake
    NAMESPACE SMAX::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)

# Generate config files
include(CMakePackageConfigHelpers)

# Ensure the Config.cmake.in template exists in the cmake/ folder.
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SmaxKernelsConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)

# Install the generated config file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SmaxKernelsConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)