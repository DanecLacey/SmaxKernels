cmake_minimum_required(VERSION 3.22.1)

set(PROJECT_NAME "SmaxKernelsSimpleExamples")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(SmaxKernels REQUIRED)


if(NOT "${SMAX_GPU_LANG}" STREQUAL "OFF")
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${SMAX_GPU_LANG})
endif()

add_executable(main main.cpp)

# Link SmaxKernels - all dependencies should be automatically propagated
target_link_libraries(main PRIVATE SMAX::smaxkernels)
