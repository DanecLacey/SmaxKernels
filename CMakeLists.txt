# Specify the minimum required CMake version
cmake_minimum_required(VERSION 3.23.1)

# Set the project and library name variables
set(PROJECT_NAME "SmaxKernels")
set(LIBRARY_NAME "smaxkernels")

# Define the project with version and language support
project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

# Set C++ standard version
set(CPP_VERSION c++17 CACHE STRING "C++ version")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# List of source files for building the library
set(SOURCES
    src/interface.cpp
    src/error_handler.cpp
)

# List of header files for installation and IDE support
set(HEADERS
    src/interface.hpp
    src/common.hpp
    src/macros.hpp
    src/memory_utils.hpp
    src/stopwatch.hpp
    src/kernel.hpp
    src/error_handler.hpp
    src/utils.hpp
)

# Additional directories containing public headers (subfolders to install)
set(HEADER_DIRS
    src/kernels
    src/utils
)

# Default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
# Provide dropdown-like options for build types in GUIs like CMake GUI
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Debug-specific configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_LEVEL 1 CACHE STRING "Set the debug verbosity level (1, 2, or 3)")
    # Validate that debug level is within accepted values
    if(NOT DEBUG_LEVEL MATCHES "^[123]$")
        message(FATAL_ERROR "DEBUG_LEVEL must be 1, 2, or 3 (got: ${DEBUG_LEVEL})")
    endif()
    set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS 1 2 3)
    message(STATUS "Debug Level: ${DEBUG_LEVEL}")
else()
    # If not Debug, remove it from GUI (cache)
    unset(DEBUG_LEVEL CACHE)
endif()

# Default to g++ if no compiler is specified
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "g++" CACHE STRING "Default compiler" FORCE)
endif()

# Display chosen compiler info
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")

# Normalize compiler ID string to uppercase
string(TOUPPER "${CMAKE_CXX_COMPILER_ID}" COMPILER_ID)

# Set compiler flags based on compiler type and build type
if(COMPILER_ID STREQUAL "GNU")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fstrict-aliasing -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
elseif(COMPILER_ID STREQUAL "INTEL")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-alias -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
elseif(COMPILER_ID STREQUAL "INTELLLVM")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fno-omit-frame-pointer -fp-model=precise")
    endif()
elseif(COMPILER_ID STREQUAL "CLANG")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Set internal debug mode variable for preprocessing logic
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(SMAX_DEBUG_MODE 1)
else()
  set(SMAX_DEBUG_MODE 0)
endif()

# Configure and generate a configuration header
set(SMAX_CONFIG_IN ${CMAKE_CURRENT_SOURCE_DIR}/cmake/smax_config.hpp.in)
set(SMAX_CONFIG_OUT ${CMAKE_CURRENT_BINARY_DIR}/include/smax_config.hpp)

# Reconfigure if input file or build type changes
set_property(
    DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS 
    ${SMAX_CONFIG_IN} "${CMAKE_CURRENT_LIST_FILE}"
)

# Process the config header template
configure_file(${SMAX_CONFIG_IN} ${SMAX_CONFIG_OUT} @ONLY)

# Create the shared library target
add_library(${LIBRARY_NAME} SHARED ${SOURCES})

# Add DEBUG_LEVEL definition to the library only in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(${LIBRARY_NAME}
    PRIVATE
      DEBUG_LEVEL=${DEBUG_LEVEL}
  )
endif()

# Include directories used when building or installing the library
target_include_directories(${LIBRARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

# Build configuration options
option(USE_TIMERS "Enable Timers support" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_LIKWID "Enable LIKWID support" OFF)

# Enable Timers if selected
if(USE_TIMERS)
    target_compile_definitions(${LIBRARY_NAME} PUBLIC USE_TIMERS)
    message(STATUS "Enabling Timers support.")
endif()

# Enable and configure OpenMP support
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found, enabling OpenMP support.")
        set(OpenMP_COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message(FATAL_ERROR "OpenMP requested but not found.")
    endif()
else()
    message(STATUS "OpenMP support is disabled.")
endif()

# Enable and configure LIKWID support
if(USE_LIKWID)
    find_package(likwid REQUIRED)
    if(likwid_FOUND)
        message(STATUS "LIKWID found, enabling LIKWID support.")
        target_compile_definitions(${LIBRARY_NAME} PUBLIC USE_LIKWID LIKWID_PERFMON)
        include_directories(${LIKWID_INCLUDES})
        set(LIKWID_LIBRARIES_TO_LINK ${LIKWID_LIBRARIES})
    else()
        message(FATAL_ERROR "LIKWID requested but not found.")
    endif()
else()
    message(STATUS "LIKWID support is disabled.")
endif()

# Set default install paths (lib, include, etc.)
include(GNUInstallDirs)

# Install the compiled library to appropriate system directories
install(TARGETS ${LIBRARY_NAME}
    EXPORT SmaxKernelsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_RUNDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install public header files and generated config header
install(
  FILES
    ${HEADERS}
    ${CMAKE_CURRENT_BINARY_DIR}/include/smax_config.hpp
  DESTINATION include/${PROJECT_NAME}
)

# Install any additional header subdirectories (e.g. utils, kernels)
install(DIRECTORY ${HEADER_DIRS} DESTINATION include/${PROJECT_NAME})

# Export the target definitions for reuse in other CMake projects
install(EXPORT SmaxKernelsTargets
    FILE SmaxKernelsTargets.cmake
    NAMESPACE SMAX::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)

# Generate a config file for find_package support
include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SmaxKernelsConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)

# Install the generated config file to the package config directory
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SmaxKernelsConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)
