# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 Lars Melchior and contributors

set(CPM_DOWNLOAD_VERSION 0.40.8)
set(CPM_HASH_SUM "78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791")

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

file(DOWNLOAD
     https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
     ${CPM_DOWNLOAD_LOCATION} 
     EXPECTED_HASH SHA256=${CPM_HASH_SUM}
     TIMEOUT 5
     STATUS DOWNLOAD_STATUS
     LOG DOWNLOAD_LOG
)

list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
list(GET DOWNLOAD_STATUS 1 DOWNLOAD_MESSAGE)

if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to download CPM.cmake (code: ${DOWNLOAD_RESULT}): ${DOWNLOAD_MESSAGE}\nLog: ${DOWNLOAD_LOG}")
endif()

set(FETCHCONTENT_QUIET FALSE)

include(${CPM_DOWNLOAD_LOCATION})

get_cmake_property(_cache_vars CACHE_VARIABLES)
foreach(var ${_cache_vars})
    string(FIND "${var}" "CPM_" CPM_POS)
    string(FIND "${var}" "FETCHCONTENT_" FC_POS)

    if(CPM_POS EQUAL 0 OR FC_POS EQUAL 0)
        mark_as_advanced(${var})
    endif()
endforeach()
