add_executable(gauss_seidel_app gauss_seidel.cpp)

# Link against the exported SMAX target
target_link_libraries(gauss_seidel_app PRIVATE SMAX::smaxkernels)

# Register the target for parent CMakeLists
set(gauss_seidel_targets gauss_seidel_app PARENT_SCOPE)