cmake_minimum_required(VERSION 3.22.1)

set(PROJECT_NAME "SmaxKernelsExamples")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

# Set global output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(SmaxKernels REQUIRED)

# Set C++ standard version
set(CPP_VERSION c++17 CACHE STRING "C++ version")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If the lib is built with CUDA, the examples must also be
if(SMAX_CUDA_MODE_EXPORT) # Detected from smax lib
    enable_language(CUDA)
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86) # Only NVIDIA Ampere supported for now 
    endif()
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(CUDA_MODE)
    include_directories(${CUDAToolkit_INCLUDE_DIRS}) # Why is this necessary?
else()
    # The examples cannot be built with CUDA if the lib was not
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        message(WARNING "CUDA compiler found, but SMAX_CUDA_MODE_EXPORT=${SMAX_CUDA_MODE_EXPORT}.
        Building only CPU examples.")
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
    add_compile_definitions(DEBUG_MODE)
    add_compile_definitions(DEBUG_LEVEL=${DEBUG_LEVEL})
    message(STATUS "Debug Level: ${DEBUG_LEVEL}")
endif()

# If the user didn't specify a compiler, default to g++
if(NOT CMAKE_CXX_COMPILER)
    message(STATUS "No C++ compiler set explicitly. Falling back to g++.")
    set(CMAKE_CXX_COMPILER "g++")
endif()

# Display the selected compiler
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
if (SMAX_CUDA_MODE_EXPORT)
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# Options for external library support
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_MPI "Enable MPI support" OFF)
option(USE_LIKWID "Enable LIKWID support" OFF)
option(USE_EIGEN "Enable Eigen support" OFF)
option(USE_MKL "Enable MKL support" OFF)
option(USE_AVX512 "Enable IntelLLVM-AVX512 support" OFF)
option(USE_CUSPARSE "Enable CUSPARSE support" OFF)
if(USE_MKL)
    add_compile_definitions(USE_MKL)
    # Option to select MKL integer interface:
    #   OFF = LP64 (32-bit indices)
    #   ON  = ILP64 (64-bit indices)
    option(MKL_USE_ILP64 "Use MKL ILP64 interface (64-bit indices)" OFF)
endif()
option(USE_FAST_MMIO "Enable fast matrix market support" OFF)
include(cmake/CPM.cmake)

# Compiler-specific flags
# GPU builds
if(SMAX_CUDA_MODE_EXPORT)
    string(TOUPPER "${CMAKE_CUDA_COMPILER_ID}" GPU_COMPILER_ID) # Normalize compiler ID for consistent matching
    if(GPU_COMPILER_ID STREQUAL "NVIDIA" OR GPU_COMPILER_ID STREQUAL "NVHPC")
        set(CMAKE_POSITION_INDEPENDENT_CODE ON) # better way to handle -Xcompiler=-fPIC
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
        elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
        endif()
    else()
        message(FATAL_ERROR "No supported GPU compiler found, even though SMAX_CUDA_MODE_EXPORT=${SMAX_CUDA_MODE_EXPORT}.")
    endif()
    message(STATUS "CUDA Threads per Block: ${CUDA_TPB}")
    message(STATUS "CUDA Architecture set to ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA compiler flags set to ${CMAKE_CUDA_FLAGS} ")
endif()

# CPU builds (default)
set(CXX_DEBUG_BASE_FLAGS "-g -O0 -Wall -Wextra -fno-omit-frame-pointer")
set(CXX_DEBUG_EXTRA_FLAGS "")

set(CXX_RELEASE_BASE_FLAGS "-O3")
set(CXX_RELEASE_EXTRA_FLAGS "")

# CPU builds (default)
list(APPEND HEADER_DIRS ${CPU_HEADER_DIRS})
string(TOUPPER "${CMAKE_CXX_COMPILER_ID}" CPU_COMPILER_ID) # Normalize compiler ID for consistent matching
if(CPU_COMPILER_ID STREQUAL "GNU")
    set(CXX_RELEASE_EXTRA_FLAGS "-march=native -fstrict-aliasing")
elseif(CPU_COMPILER_ID STREQUAL "INTEL")
    set(CXX_RELEASE_EXTRA_FLAGS "-march=native -fno-alias")
elseif(CPU_COMPILER_ID STREQUAL "INTELLLVM")
    if(USE_AVX512)
        set(CXX_RELEASE_EXTRA_FLAGS "-xHost -qopt-zmm-usage=high")
    else()
        set(CXX_RELEASE_EXTRA_FLAGS "-xHost")
    endif()
    set(CXX_DEBUG_EXTRA_FLAGS "-fp-model=precise")
elseif(CPU_COMPILER_ID STREQUAL "CLANG")
    set(CXX_RELEASE_EXTRA_FLAGS "-fno-strict-aliasing")
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_DEBUG_BASE_FLAGS} ${CXX_DEBUG_EXTRA_FLAGS}")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_RELEASE_BASE_FLAGS} ${CXX_RELEASE_EXTRA_FLAGS}")
endif()

message(STATUS "CXX FLAGS USED : ${CMAKE_CXX_FLAGS}")

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
if (USE_LIKWID OR SMAX_LIKWID_MODE_EXPORT)
    if (NOT USE_LIKWID)
        message(WARNING "USE_LIKWID=OFF but SMAX_LIKWID_MODE_EXPORT=ON; enabling LIKWID support anyway.")
    else()
        message(STATUS "LIKWID support explicitly enabled via USE_LIKWID=ON.")
    endif()

    find_package(likwid REQUIRED)

    if (likwid_FOUND)
        message(STATUS "LIKWID found, enabling LIKWID support.")
    else()
        message(FATAL_ERROR "LIKWID requested but not found.")
    endif()

    add_compile_definitions(USE_LIKWID LIKWID_PERFMON)
    include_directories(${LIKWID_INCLUDES})
    set(LIKWID_LIBRARIES_TO_LINK ${LIKWID_LIBRARIES} CACHE INTERNAL "Libraries for LIKWID")
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

else()
    message(STATUS "Eigen support is disabled.")
endif()

# Enable MKL support
if(USE_MKL)
    if(MKL_USE_ILP64)
        if(SMAX_RACE_MODE_EXPORT)
            message(FATAL_ERROR "MKL_USE_ILP64 cannot be used with SMAX_RACE_MODE at the moment .")
        endif()
        set(MKL_INTERFACE ilp64)
        add_compile_definitions(USE_MKL_ILP64)
    else()
        set(MKL_INTERFACE lp64)
        add_compile_definitions(USE_MKL_LP64)
    endif()

    find_package(MKL REQUIRED)
    if(MKL_FOUND)
        message(STATUS "MKL found, enabling MKL support.")
        if(MKL_USE_ILP64)
            message(STATUS "Building MKL with ILP64 (64-bit indices)")
        else()
            message(STATUS "Building MKL with LP64 (32-bit indices)")
        endif()
    else()
        message(FATAL_ERROR "MKL requested but not found.")
    endif()
else()
    message(STATUS "MKL support is disabled.")
endif()

set(VERBOSITY 1 CACHE STRING "Set the validation verbosity level (0 or 1)")
if(NOT VERBOSITY MATCHES "^[01]$")
    message(FATAL_ERROR "VERBOSITY must be 0 or 1 (got: ${VERBOSITY})")
endif()
set_property(CACHE VERBOSITY PROPERTY STRINGS 0 1)
message(STATUS "Validation verbosity level: ${VERBOSITY}")

if(NOT (USE_MKL OR USE_CUSPARSE))
    message(WARNING "No validation back‐ends enabled (set USE_MKL or USE_CUSPARSE). Skipping validation.")
endif()

set(validation_targets)

# MKL validation
if(USE_MKL)
    message(STATUS "Configuring MKL validation")
    add_subdirectory(validation/mkl)
    list(APPEND validation_targets ${mkl_validation_targets})

    if(NOT CPU_COMPILER_ID STREQUAL "INTELLLVM" AND NOT CPU_COMPILER_ID STREQUAL "INTEL")
        message(WARNING "MKL validation built without an Intel compiler. Expect possible ABI issues.")
    endif()
endif()

# cuSPARSE validation
if(USE_CUSPARSE)
    message(STATUS "Configuring cuSPARSE validation")
    add_subdirectory(validation/cusparse)
    list(APPEND validation_targets ${cusparse_validation_targets})
endif()

# Finally build a combined validation target
add_custom_target(validation
    DEPENDS ${validation_targets}
    COMMENT "Building all enabled validation checks"
)

# === fast mmio Support ===
if(USE_FAST_MMIO)
    message(STATUS "Fast MMIO support is enabled.")
    include(cmake/fastmmio.cmake) 
else()
    message(STATUS "Fast MMIO support is disabled.")
endif()
# sort of ugly work around 
if(USE_FAST_MMIO)
    target_sources(fmmio
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/mmio.cpp
    )
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