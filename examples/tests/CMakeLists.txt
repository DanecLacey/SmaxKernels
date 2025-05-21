# Find all test headers
file(GLOB TEST_HEADERS *.hpp)

# Prepare include lines
set(INCLUDE_HEADERS)
foreach(header ${TEST_HEADERS})
    get_filename_component(name ${header} NAME)
    list(APPEND INCLUDE_HEADERS "#include \"${name}\"")
endforeach()

# Join include lines into a single string with newlines
string(REPLACE ";" "\n" INCLUDE_HEADERS "${INCLUDE_HEADERS}")

# Configure run_tests.cpp from template
configure_file(
    run_tests.cpp.in
    run_tests.cpp
    @ONLY
)

# Build the test runner
add_executable(run_tests run_tests.cpp)
target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(run_tests PRIVATE SMAX::smaxkernels)
set(test_targets run_tests PARENT_SCOPE)
