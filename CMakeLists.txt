CMAKE_MINIMUM_REQUIRED (VERSION 3.23.1)

################################################################################
### Setting Global Parameters

SET(PROJECT_NAME "SmaxKernels")
SET(LIBRARY_NAME "smaxkernels")

PROJECT("${PROJECT_NAME}")

SET(LIBRARY_VERSION_MAJOR 0)
SET(LIBRARY_VERSION_STRING 0.1)

################################################################################
### Project definition

SET(SOURCES
	"src/interface.cpp"
    "src/error_handler.cpp"
)

SET(HEADERS
	"src/interface.hpp"
    "src/common.hpp"
    "src/macros.hpp"
    "src/memory_utils.hpp"
    "src/stopwatch.hpp"
    "src/kernel.hpp"
    "src/error_handler.hpp"
)

SET(HEADER_DIRS
    "src/kernels"
)

ADD_LIBRARY (${LIBRARY_NAME} SHARED ${SOURCES})

################################################################################
### Adding compilator definitions

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (e.g., Debug, Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_LEVEL 1 CACHE STRING "Set the debug level (1=low, 2=medium, 3=high)")
endif()


message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Check if CXX is set, if not, set it to the default (e.g., g++)
if(NOT DEFINED CMAKE_CXX_COMPILER)
    if(DEFINED ENV{CXX})
        set(CMAKE_CXX_COMPILER $ENV{CXX})
    else()
        set(CMAKE_CXX_COMPILER "g++")  # Or another default, if preferred
    endif()
endif()

# CMake automatically detects the compiler ID once CMAKE_CXX_COMPILER is set.
message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
    message(STATUS "CXX Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    # Default
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

    # TODO: More arch specific flags
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -O3")
        endif()
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            set(CMAKE_CXX_FLAGS "-DDEBUG_MODE  -DDEBUG_LEVEL=${DEBUG_LEVEL} ${CMAKE_CXX_FLAGS} -g -Wall -Wextra")
        endif()
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Profile")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
        endif()
    endif()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias -O3")
        endif()
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            set(CMAKE_CXX_FLAGS "-DDEBUG_MODE -DDEBUG_LEVEL=${DEBUG_LEVEL} ${CMAKE_CXX_FLAGS} -g -Wall -Wextra")
        endif()
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Profile")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
        endif()
    endif()
else()
    message(FATAL_ERROR "Compiler not recognized.")
endif()

################################################################################
### Options

# Option for enabling Timers
option(USE_TIMERS "Enable Timers support" ON)
if(USE_TIMERS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TIMERS")
endif()

# Option for enabling OpenMP
option(USE_OPENMP "Enable OpenMP support" ON)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found, enabling OpenMP support.")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message(FATAL_ERROR "OpenMP requested but not found.")
    endif()
else()
    message(STATUS "OpenMP support is disabled.")
endif()

# Option for enabling MPI
option(USE_MPI "Enable MPI support" OFF)

if(USE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found, enabling MPI support.")
        # Use the MPI wrapper around the compiler
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_COMPILER "mpic++")
        endif()
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
            set(CMAKE_CXX_COMPILER "mpiicpc")
        endif()
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
            set(CMAKE_CXX_COMPILER "mpiicpc -cxx=icpx")
        endif()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS} -DUSE_MPI")
    else()
        message(FATAL_ERROR "MPI requested but not found.")
    endif()
else()
    message(STATUS "MPI support is disabled.")
endif()

# option(GRAPH_TOPO "Enable Graph Topology MPI Communicator" ON)
# if(GRAPH_TOPO)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGRAPH_TOPO")
# endif()

# Option for enabling RACE
# option(USE_RACE "Enable RACE support (cache blocking)" OFF)

# if(USE_RACE)
#     set(RACE_LIB_DIR "" CACHE PATH "Path to the RACE library directory")

#     if(RACE_LIB_DIR)
#         message(STATUS "RACE found, enabling RACE support (permutations).")
#         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_RACE")

#         set(HWLOC_LIB_DIR "" CACHE PATH "Path to the HWLOC library directory")
#         if(HWLOC_LIB_DIR)
#             message(STATUS "HWLOC found, enabling HWLOC support.")
#             # Include HWLOC headers
#             target_include_directories(${LIBRARY_NAME} PUBLIC ${HWLOC_LIB_DIR}/include)

#             # Allow linker to find HWLOC library
#             target_link_directories(${LIBRARY_NAME} PUBLIC ${HWLOC_LIB_DIR}/lib)

#             # Link against HWLOC
#             target_link_libraries(${LIBRARY_NAME} PUBLIC libhwloc.so)

#         else()
#             message(FATAL_ERROR "RACE requested, but HWLOC dependency was not found.")
#         endif()

#         # Include RACE headers
#         target_include_directories(${LIBRARY_NAME} PUBLIC ${RACE_LIB_DIR}/include)

#         # Allow linker to find RACE library
#         target_link_directories(${LIBRARY_NAME} PUBLIC ${RACE_LIB_DIR}/lib/RACE)
        
#         # Link against RACE
#         target_link_libraries(${LIBRARY_NAME} PUBLIC libRACE.so)

#     else()
#         message(FATAL_ERROR "RACE requested but not found.")
#     endif()
# else()
#     message(STATUS "RACE support is disabled.")
# endif()

# Option for enabling LIKWID
# option(USE_LIKWID "Enable likwid support" OFF)

# if(USE_LIKWID)
#     find_package(likwid REQUIRED)
#     if (likwid_FOUND)
#         message(STATUS "likwid found, enabling likwid support.")

#         target_link_libraries(${LIBRARY_NAME} PUBLIC ${LIKWID_LIBRARIES})
#         target_include_directories(${LIBRARY_NAME} PUBLIC ${LIKWID_INCLUDES})

# 		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIKWID_PERFMON -DUSE_LIKWID")
#     else()
#         message(FATAL_ERROR "likwid requested but not found.")
#     endif()
# else()
#     message(STATUS "likwid support is disabled.")
# endif()

################################################################################
### Library installation directives
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

SET(INCLUDE_DIR include/${PROJECT_NAME} CACHE PATH "Path to include dir")

# Install the header files
install(FILES ${HEADERS} DESTINATION ${INCLUDE_DIR})
install(DIRECTORY ${HEADER_DIRS} DESTINATION ${INCLUDE_DIR})

INSTALL (
	TARGETS ${LIBRARY_NAME}
	LIBRARY DESTINATION lib
	INCLUDES DESTINATION include
)

################################################################################
### Build tests
add_subdirectory(tests)

# Create a top-level "tests" target
add_custom_target(tests
    COMMENT "Build all test executables"
)
foreach(tgt ${TEST_TARGETS})
    add_dependencies(tests ${tgt})
endforeach()

# ################################################################################
### Build benchmarks
add_subdirectory(benchmarks)

add_custom_target(benchmarks
    COMMENT "Build all benchmark targets"
)

foreach(benchmark_target IN LISTS BENCHMARK_TARGETS)
    add_dependencies(benchmarks ${benchmark_target})
endforeach()

# ################################################################################
### Build applications
add_subdirectory(applications)

add_custom_target(applications
    COMMENT "Build all application targets"
)

foreach(application_target IN LISTS APPLICATION_TARGETS)
    add_dependencies(applications ${application_target})
endforeach()

# ################################################################################