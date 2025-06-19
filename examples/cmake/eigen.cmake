CPMAddPackage(
  NAME Eigen  
  VERSION 3.4.0 
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz  
  DOWNLOAD_ONLY YES  
)

if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)  
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
  add_library(Eigen3::Eigen ALIAS Eigen)
endif()