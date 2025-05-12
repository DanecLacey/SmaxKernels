add_executable(jacobi jacobi.cpp)

# Link against the exported SMAX target
target_link_libraries(jacobi PRIVATE SMAX::smaxkernels)

# Register the target for parent CMakeLists
set(jacobi_targets jacobi PARENT_SCOPE)