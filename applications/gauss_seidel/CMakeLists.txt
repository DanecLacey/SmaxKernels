add_executable(gauss_seidel gauss_seidel.cpp)

# Tell the compiler where to find the headers
target_include_directories(gauss_seidel PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include/
)

target_link_libraries(gauss_seidel PRIVATE ${LIBRARY_NAME})

set(gauss_seidel_TARGETS gauss_seidel)

# Propagate back to the parent scope
set(gauss_seidel_TARGETS ${gauss_seidel_TARGETS} PARENT_SCOPE)