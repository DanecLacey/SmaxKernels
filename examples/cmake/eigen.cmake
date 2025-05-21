# Use CPM to fetch and manage the Eigen library
CPMAddPackage(
  NAME Eigen  # Package name
  VERSION 3.2.8  # Specify Eigen version
  URL https://gitlab.com/libeigen/eigen/-/archive/3.2.8/eigen-3.2.8.tar.gz  # Download URL for Eigen

  # Eigen does not provide a standard CMakeLists.txt for building as a library
  DOWNLOAD_ONLY YES  # Only download Eigen, do not attempt to build it
)

# If Eigen was successfully added, define it as an imported interface library
if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)  # Create an imported interface target for Eigen
    #add_library(Eigen3::Eigen ALIAS Eigen)
  # Set Eigen's include directories so that projects using Eigen can find its headers
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()