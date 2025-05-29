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

# Optional: restrict required version
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86)
    endif()

    enable_language(CUDA)
endif()

# List of source files for building the library
set(SOURCES
    src/interface.cpp
    src/error_handler.cpp
)

# List of header files for installation
set(COMMON_HEADERS
    src/interface.hpp
    src/common.hpp
    src/macros.hpp
    src/memory_utils.hpp
    src/stopwatch.hpp
    src/kernel.hpp
    src/error_handler.hpp
    src/utils.hpp
)

set(KERNEL_HEADERS
    src/kernels/kernels_common.hpp
    src/kernels/spgemm.hpp
    src/kernels/spgemm/spgemm_common.hpp
    src/kernels/spgemm/spgemm_cpu.hpp
    src/kernels/spmm.hpp
    src/kernels/spmm/spmm_common.hpp
    src/kernels/spmm/spmm_cpu.hpp
    src/kernels/spmv.hpp
    src/kernels/spmv/spmv_common.hpp
    src/kernels/spmv/spmv_cpu.hpp
    src/kernels/spmv/spmv_cuda.hpp
    src/kernels/sptrsm.hpp
    src/kernels/sptrsm/sptrsm_common.hpp
    src/kernels/sptrsm/sptrsm_cpu.hpp
    src/kernels/sptrsv.hpp
    src/kernels/sptrsv/sptrsv_common.hpp
    src/kernels/sptrsv/sptrsv_cpu.hpp
)

set(CPU_HEADER_DIRS
    src/kernels/spgemm/spgemm_cpu/
    src/kernels/spmm/spmm_cpu/
    src/kernels/spmv/spmv_cpu/
    src/kernels/sptrsm/sptrsm_cpu/
    src/kernels/sptrsv/sptrsv_cpu/
    src/platforms/cpu/
)

set(CUDA_HEADER_DIRS
    src/kernels/spmv/spmv_cuda/
    src/platforms/cuda
)

set(COMMON_HEADER_DIRS
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
    set(DEBUG_MODE 1)
    set(DEBUG_LEVEL 1 CACHE STRING "Set the debug verbosity level (1, 2, or 3)")
    # Validate that debug level is within accepted values
    if(NOT DEBUG_LEVEL MATCHES "^[123]$")
        message(FATAL_ERROR "DEBUG_LEVEL must be 1, 2, or 3 (got: ${DEBUG_LEVEL})")
    endif()
    set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS 1 2 3)
    message(STATUS "Debug Level: ${DEBUG_LEVEL}")
else()
    set(DEBUG_MODE 0)
    # If not Debug, remove it from GUI cache
    unset(DEBUG_LEVEL CACHE)
endif()

# If the user didn't specify a compiler, default to g++
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "g++" CACHE STRING "Default compiler" FORCE)
    message(STATUS "No compiler selected, using default: ${CMAKE_CXX_COMPILER}")
endif()


# Display chosen compiler info
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# Normalize compiler ID string to uppercase
string(TOUPPER "${CMAKE_CXX_COMPILER_ID}" CPU_COMPILER_ID)
string(TOUPPER "${CMAKE_CUDA_COMPILER_ID}" GPU_COMPILER_ID)

# Set compiler flags based on compiler type and build type
set(HEADERS ${COMMON_HEADERS} ${KERNEL_HEADERS})
set(HEADER_DIRS ${COMMON_HEADER_DIRS})

# GPU builds
if(GPU_COMPILER_ID STREQUAL "NVIDIA" OR GPU_COMPILER_ID STREQUAL "NVHPC")
    set(USE_CUDA 1)
    message(STATUS "CUDA Architecture set to ${CMAKE_CUDA_ARCHITECTURES}")
    # Set CUDA flags based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=-fPIC")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -Xcompiler=-fPIC")
    endif()

    # Include CUDA headers
    list(APPEND HEADER_DIRS ${CUDA_HEADER_DIRS})

    # Set CUDA threads-per-block if not already defined
    if(NOT DEFINED CUDA_TPB)
        set(CUDA_TPB 32 CACHE STRING "CUDA Threads Per Block")
    endif()
    add_compile_definitions(CUDA_TPB=${CUDA_TPB})
    message(STATUS "CUDA Threads per Block: ${CUDA_TPB}")

    # Add CUDA source file
    list(APPEND SOURCES
        src/kernels/spmv/spmv_cuda/spmv_cuda_crs_impl.cu
    )
else()
    set(USE_CUDA 0)
    message(WARNING "No supported GPU compiler found, building for CPU only.")
endif()

set(CXX_DEBUG_BASE_FLAGS "-g -O0 -Wall -Wextra -fno-omit-frame-pointer")
set(CXX_DEBUG_EXTRA_FLAGS "")

set(CXX_RELEASE_BASE_FLAGS "-O3")
set(CXX_RELEASE_EXTRA_FLAGS "")

# CPU builds (default)
list(APPEND HEADER_DIRS ${CPU_HEADER_DIRS})
if(CPU_COMPILER_ID STREQUAL "GNU")
    set(CXX_RELEASE_EXTRA_FLAGS "-march=native -fstrict-aliasing")
elseif(CPU_COMPILER_ID STREQUAL "INTEL")
    set(CXX_RELEASE_EXTRA_FLAGS "-march=native -fno-alias")
elseif(CPU_COMPILER_ID STREQUAL "INTELLLVM")
    set(CXX_RELEASE_EXTRA_FLAGS "-xHost -O3")
    set(CXX_DEBUG_EXTRA_FLAGS "-fp-model=precise")
elseif(CPU_COMPILER_ID STREQUAL "CLANG")
    set(CXX_RELEASE_EXTRA_FLAGS "-fno-alias")
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_DEBUG_BASE_FLAGS} ${CXX_DEBUG_EXTRA_FLAGS}")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_RELEASE_BASE_FLAGS} ${CXX_RELEASE_EXTRA_FLAGS}")
endif()

message(STATUS "CXX FLAGS USED : ${CMAKE_CXX_FLAGS}")

# Create the shared library target
add_library(${LIBRARY_NAME} SHARED ${SOURCES})

target_compile_definitions(${LIBRARY_NAME} PUBLIC
    USE_CUDA=${USE_CUDA}
    DEBUG_MODE=${DEBUG_MODE}
)

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

# Add DEBUG_LEVEL definition to the library only in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${LIBRARY_NAME}
        PRIVATE DEBUG_LEVEL=${DEBUG_LEVEL})
endif()

# Include directories used when building or installing the library
target_include_directories(${LIBRARY_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)


if(CUDAToolkit_FOUND)
    # This tells the host compiler where to find CUDA headers
    target_include_directories(${LIBRARY_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # Propagate to users
    target_compile_definitions(${LIBRARY_NAME} PUBLIC CUDA_TPB=${CUDA_TPB})
    target_include_directories(${LIBRARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} PUBLIC CUDA::cudart)

    # Mark .cu files as CUDA language explicitly (optional but safe)
    set_source_files_properties(
        src/kernels/spmv/spmv_cuda/spmv_cuda_crs_impl.cu
        PROPERTIES LANGUAGE CUDA
    )
endif()

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
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path prefix")

include(GNUInstallDirs)

# Install the compiled library to appropriate system directories
install(TARGETS ${LIBRARY_NAME}
    EXPORT SmaxKernelsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_RUNDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install generated config header
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/smax_config.hpp
    DESTINATION include/${PROJECT_NAME}
)

# Manually forcing header installations to maintain desired dir structure
set(SRC_ROOT_DIR "${CMAKE_SOURCE_DIR}/src")

foreach(base_dir IN LISTS HEADER_DIRS)
    get_filename_component(base_dir_abs "${base_dir}" ABSOLUTE)

    file(GLOB_RECURSE dir_headers CONFIGURE_DEPENDS
        "${base_dir_abs}/*.hpp"
        "${base_dir_abs}/*.cuh"
    )
    foreach(header IN LISTS dir_headers)
        get_filename_component(header_abs "${header}" ABSOLUTE)
        # Use SRC_ROOT_DIR instead of base_dir_abs here!
        file(RELATIVE_PATH rel_path "${SRC_ROOT_DIR}" "${header_abs}")
        get_filename_component(rel_dir "${rel_path}" DIRECTORY)
        install(FILES "${header_abs}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${rel_dir}")
    endforeach()
endforeach()

foreach(header IN LISTS HEADERS)
    get_filename_component(header_abs "${header}" ABSOLUTE)
    file(RELATIVE_PATH rel_path "${SRC_ROOT_DIR}" "${header_abs}")
    get_filename_component(rel_dir "${rel_path}" DIRECTORY)
    install(FILES "${header_abs}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${rel_dir}")
endforeach()


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
