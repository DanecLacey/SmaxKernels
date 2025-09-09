cmake_minimum_required(VERSION 3.22.1)

set(PROJECT_NAME "SmaxKernelsSimpleExamples")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

# Set global output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find SmaxKernels - this will automatically set up all dependencies
find_package(SmaxKernels REQUIRED)

# Set C++ standard version
set(CPP_VERSION c++17 CACHE STRING "C++ version")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable GPU language if SmaxKernels was built with GPU support
if(NOT "${SMAX_GPU_LANG}" STREQUAL "OFF")
    enable_language(${SMAX_GPU_LANG})
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${SMAX_GPU_LANG})
endif()

add_executable(main main.cpp)

# Link SmaxKernels - all dependencies should be automatically propagated
target_link_libraries(main PUBLIC SMAX::smaxkernels)
