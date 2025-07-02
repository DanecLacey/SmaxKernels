add_executable(jacobi_app jacobi.cpp)

# Link against the exported SMAX target
target_link_libraries(jacobi_app PRIVATE SMAX::smaxkernels)

# Register the target for parent CMakeLists
set(jacobi_targets jacobi_app PARENT_SCOPE)