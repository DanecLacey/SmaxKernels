cmake_minimum_required(VERSION 3.22.1)

set(PROJECT_NAME "SmaxKernelsSimpleExamples")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

# Set global output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(SmaxKernels REQUIRED)

# Set C++ standard version
set(CPP_VERSION c++17 CACHE STRING "C++ version")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

find_package(OpenMP)
find_package(CUDAToolkit)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC SMAX::smaxkernels)
