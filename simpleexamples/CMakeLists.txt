cmake_minimum_required(VERSION 3.22)
project(SmaxKernelsExamples)

enable_language(HIP)
find_package(SmaxKernels REQUIRED)
find_package(OpenMP REQUIRED)

find_package(CUDAToolkit)
add_executable(main main.cpp)
target_link_libraries(main PRIVATE SMAX::smaxkernels)
target_link_libraries(main PRIVATE CUDA::cudart)
target_compile_definitions(main PRIVATE __HIP_PLATFORM_NVIDIA__)
