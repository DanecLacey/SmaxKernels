cmake_minimum_required(VERSION 3.22.1)

set(PROJECT_NAME "SmaxKernelsSimpleExamples")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

# Set global output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(SmaxKernels REQUIRED)

# Set C++ standard version
set(CPP_VERSION c++17 CACHE STRING "C++ version")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main main.cpp)
target_link_libraries(main )