cmake_minimum_required(VERSION 3.23.1)

set(PROJECT_NAME "SmaxKernelsExamples")

project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

find_package(SmaxKernels REQUIRED)

add_subdirectory(tests)
add_subdirectory(applications)