cmake_minimum_required(VERSION 3.23.1)

set(PROJECT_NAME "SmaxKernelsExamples")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

find_package(SmaxKernels REQUIRED)

# Set C++ standard version
set(CPP_VERSION c++17 CACHE STRING "C++ version")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If the lib is built with CUDA, the examples must also be
if(SMAX_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
else()
    # The examples can be built with CUDA, even if the lib was not
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 80 86)
        endif()
        if(NOT SMAX_USE_CUDA)
            message(WARNING
                "CUDA compiler found, but SMAX_USE_CUDA=${SMAX_USE_CUDA}. "
                "CUDA examples will not be built.")
        endif()
    endif()
endif()

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
    message(STATUS "No compiler selected, using default: ${CMAKE_CXX_COMPILER}")
endif()

# Display the selected compiler
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# Normalize compiler ID for consistent matching
string(TOUPPER "${CMAKE_CXX_COMPILER_ID}" CPU_COMPILER_ID)
string(TOUPPER "${CMAKE_CUDA_COMPILER_ID}" GPU_COMPILER_ID)

# Compiler-specific flags
# GPU builds
if(GPU_COMPILER_ID STREQUAL "NVIDIA")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=-fPIC")
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -Xcompiler=-fPIC")
    endif()
else()
    message(STATUS "No supported GPU compiler found, building for CPU only.")
endif()

# CPU builds (default)
if(CPU_COMPILER_ID STREQUAL "GNU")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fstrict-aliasing -O3")
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
elseif(CPU_COMPILER_ID STREQUAL "INTEL")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-alias -O3")
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
elseif(CPU_COMPILER_ID STREQUAL "INTELLLVM")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -O3")
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fno-omit-frame-pointer -fp-model=precise")
    endif()
elseif(CPU_COMPILER_ID STREQUAL "CLANG")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias -O3")
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Options for external library support
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_MPI "Enable MPI support" OFF)
option(USE_LIKWID "Enable LIKWID support" OFF)
option(USE_EIGEN "Enable Eigen support" OFF)
option(USE_MKL "Enable MKL support" OFF)
option(USE_PETSC "Enable PETSC support" OFF)
option(USE_FAST_MMIO "Enable fast matrix market support" OFF)
include(cmake/CPM.cmake)

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
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
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
        set(LIKWID_LIBRARIES_TO_LINK ${LIKWID_LIBRARIES} CACHE INTERNAL "Libraries for LIKWID")
    else()
        message(FATAL_ERROR "LIKWID requested but not found.")
    endif()
else()
    message(STATUS "LIKWID support is disabled.")
endif()

# Enable Eigen support
if(USE_EIGEN)
    find_package(Eigen3 QUIET)

    if(Eigen3_FOUND)
        message(STATUS "Using system-installed Eigen (Eigen3::Eigen)")
    else()
        message(STATUS "Eigen not found locally, fetching via CPM")
        include(cmake/eigen.cmake)
    endif()

    # Optional: Compiler flags specific to Eigen performance tuning
    add_compile_options(-mavx2 -mno-avx512f)
else()
    message(STATUS "Eigen support is disabled.")
endif()


# Enable MKL support
if(USE_MKL)
    # Ensure we're using LP64 interface (32-bit integers) for MKL
    # TODO: This isn't explicitly necessary. Make more flexible.
    set(MKL_INTERFACE lp64)
    find_package(MKL REQUIRED)
    if(MKL_FOUND)
        message(STATUS "MKL found, enabling MKL support.")
    else()
        message(FATAL_ERROR "MKL requested but not found.")
    endif()
else()
    message(STATUS "MKL support is disabled.")
endif()

# Enable PETSC support
if(USE_PETSC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)
    if(PETSC_FOUND)
        message(STATUS "PETSc found, enabling PETSC support.")
    else()
        message(FATAL_ERROR "PETSc requested but not found.")
    endif()
else()
    message(STATUS "PETSc support is disabled.")
endif()


if(USE_MKL)
    # DL 05.04.25 TODO: Would be useful to have other intermediate levels
    set(VERBOSITY 1 CACHE STRING "Set the validation verbosity level (0 or 1)")

    # Validate debug level input
    if(NOT VERBOSITY MATCHES "^[01]$")
        message(FATAL_ERROR "VERBOSITY must be 0 or 1 (got: ${VERBOSITY})")
    endif()

    # Set the verbosity level as a compile definition
    set_property(CACHE VERBOSITY PROPERTY STRINGS 0 1)
    message(STATUS "Verbosity Level: ${VERBOSITY}")

    add_subdirectory(validation)
    add_custom_target(validation DEPENDS ${validation_targets})
else()
    message(WARNING "Validation checks not built. MKL is required.")
endif()



# === fast mmio Support ===
if(USE_FAST_MMIO)
    message(STATUS "Fast MMIO support is enabled.")
    include(cmake/fastmmio.cmake) 
else()
    message(STATUS "Fast MMIO support is disabled.")
endif()
# sort of ugly work around 
if(USE_FAST_MMIO)
    target_compile_definitions(fmmio INTERFACE USE_FAST_MMIO)
    add_library(mmio ALIAS fmmio)
else()
    add_library(mmio STATIC mmio.cpp)
    target_include_directories(mmio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_subdirectory(tests)
add_custom_target(tests DEPENDS ${test_targets})

add_subdirectory(demos)
add_custom_target(demos DEPENDS ${demos_targets})

add_subdirectory(applications)
add_custom_target(applications DEPENDS ${application_targets})

add_subdirectory(benchmarks)
add_custom_target(benchmarks DEPENDS ${benchmarks_targets})
