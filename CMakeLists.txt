cmake_minimum_required(VERSION 3.23.1)

set(PROJECT_NAME "SmaxKernels")
set(LIBRARY_NAME "smaxkernels")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

set(CPP_VERSION c++17 CACHE STRING "C++ version")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    src/utils.hpp
)

# Header directories
set(HEADER_DIRS
    src/kernels
    src/utils
)

# Set default build type if not defined
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
# Provide options for build type
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Handle Debug build-specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_LEVEL 1 CACHE STRING "Set the debug verbosity level (1, 2, or 3)")
    # Validate debug level input
    if(NOT DEBUG_LEVEL MATCHES "^[123]$")
        message(FATAL_ERROR "DEBUG_LEVEL must be 1, 2, or 3 (got: ${DEBUG_LEVEL})")
    endif()
    # Set the debug level as a compile definition
    set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS 1 2 3)
    add_compile_definitions(DEBUG_LEVEL=${DEBUG_LEVEL})
    message(STATUS "Debug Level: ${DEBUG_LEVEL}")
endif()

# If the user didn't specify a compiler, default to g++
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "g++" CACHE STRING "Default compiler" FORCE)
endif()

# Display the selected compiler
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")

# Normalize compiler ID for consistent matching
string(TOUPPER "${CMAKE_CXX_COMPILER_ID}" COMPILER_ID)

# Compiler-specific flags
if(COMPILER_ID STREQUAL "GNU")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fstrict-aliasing -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_MODE -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
elseif(COMPILER_ID STREQUAL "INTEL")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-alias -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_MODE -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
elseif(COMPILER_ID STREQUAL "INTELLLVM")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_MODE -g -O0 -Wall -Wextra -fno-omit-frame-pointer -fp-model=precise")
    endif()
elseif(COMPILER_ID STREQUAL "CLANG")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_MODE -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Define the shared library
add_library(${LIBRARY_NAME} SHARED ${SOURCES})

# Options
option(USE_TIMERS "Enable Timers support" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_LIKWID "Enable LIKWID support" OFF)

# Enable timers support if option is set
if(USE_TIMERS)
    target_compile_definitions(${LIBRARY_NAME} PUBLIC USE_TIMERS)
    message(STATUS "Enabling Timers support.")
endif()

# === OpenMP Support ===
if(USE_OPENMP)
    # Find OpenMP package and enable if found
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

# === LIKWID Support ===
if(USE_LIKWID)
    # Find LIKWID package and enable if found
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

# Include directories for the library
include(GNUInstallDirs)

# Expose headers during build and installation
target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

# Install the library to the system
install(TARGETS ${LIBRARY_NAME}
    EXPORT SmaxKernelsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_RUNDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the header files
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})
install(DIRECTORY ${HEADER_DIRS} DESTINATION include/${PROJECT_NAME})

# Install export for targets (for later use in other projects)
install(EXPORT SmaxKernelsTargets
    FILE SmaxKernelsTargets.cmake
    NAMESPACE SMAX::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)

# Generate and install config files for the package
include(CMakePackageConfigHelpers)

# Ensure the Config.cmake.in template exists in the cmake/ folder
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SmaxKernelsConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)

# Install the generated config file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SmaxKernelsConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmaxKernels
)
