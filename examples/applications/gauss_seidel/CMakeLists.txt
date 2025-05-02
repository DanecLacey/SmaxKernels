add_executable(gauss_seidel gauss_seidel.cpp)

# Link against the exported SMAX target
target_link_libraries(gauss_seidel PRIVATE SMAX::smaxkernels)

# Register the target for parent CMakeLists
set(gauss_seidel_targets gauss_seidel PARENT_SCOPE)