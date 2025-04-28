# Find all the .cpp files in the current directory and subdirectories
file(GLOB_RECURSE TEST_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

# Loop through each source file and create an executable for each
foreach(test_file ${TEST_SOURCES})
    # Get the base name of the test file (without the extension)
    get_filename_component(test_name ${test_file} NAME_WE)

    # Define an executable for each test file
    add_executable(${test_name} ${test_file})

    # Link the executable with the SmaxKernels library
    target_link_libraries(${test_name} PRIVATE SMAX::smaxkernels)

    # Optionally, set any specific compile options or include directories for testing
    target_include_directories(${test_name} PRIVATE ${SmaxKernels_INCLUDE_DIRS})
endforeach()