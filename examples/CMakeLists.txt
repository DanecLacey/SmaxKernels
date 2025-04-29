cmake_minimum_required(VERSION 3.23.1)

set(PROJECT_NAME "SmaxKernelsExamples")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

find_package(SmaxKernels REQUIRED)

add_library(mmio STATIC mmio.cpp)
target_include_directories(mmio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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

# Set default C++ compiler if not specified
if(NOT DEFINED CMAKE_CXX_COMPILER AND DEFINED ENV{CXX})
    set(CMAKE_CXX_COMPILER $ENV{CXX})
endif()

# Display the selected compiler
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")

# Set architecture flags for all builds
add_compile_options(-march=native)

# Compiler-specific flags for GNU and Intel compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "-DDEBUG_MODE  ${CMAKE_CXX_FLAGS} -g -Wall -Wextra")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "-DDEBUG_MODE  ${CMAKE_CXX_FLAGS} -g -Wall -Wextra")
    endif()
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Options for external library support
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_MPI "Enable MPI support" OFF)
option(USE_LIKWID "Enable LIKWID support" OFF)

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

# === MPI Support ===
if(USE_MPI)
    # Find MPI package and enable if found
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found, enabling MPI support.")
        add_compile_definitions(USE_MPI)
        set(MPI_COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
        set(MPI_LIBRARIES_TO_LINK ${MPI_CXX_LIBRARIES})
        include_directories(${MPI_CXX_INCLUDE_PATH})

        # Optional: switch compiler to MPI wrapper if needed
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_COMPILER "mpic++")
        elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
            set(CMAKE_CXX_COMPILER "mpiicpc")
        endif()
    else()
        message(FATAL_ERROR "MPI requested but not found.")
    endif()
else()
    message(STATUS "MPI support is disabled.")
endif()

# === LIKWID Support ===
if(USE_LIKWID)
    # Find LIKWID package and enable if found
    find_package(likwid REQUIRED)
    if(likwid_FOUND)
        message(STATUS "LIKWID found, enabling LIKWID support.")
        add_compile_definitions(USE_LIKWID LIKWID_PERFMON)
        include_directories(${LIKWID_INCLUDES})
        set(LIKWID_LIBRARIES_TO_LINK ${LIKWID_LIBRARIES})
    else()
        message(FATAL_ERROR "LIKWID requested but not found.")
    endif()
else()
    message(STATUS "LIKWID support is disabled.")
endif()

add_subdirectory(tests)
add_subdirectory(applications)
add_subdirectory(benchmarks)