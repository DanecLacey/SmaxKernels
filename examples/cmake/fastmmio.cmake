CPMAddPackage(
  NAME fmmio
  GITHUB_REPOSITORY alugowski/fast_matrix_market
  GIT_TAG v1.7.6
)

if(fmmio_ADDED)
  add_library(fmmio INTERFACE)
  target_include_directories(fmmio INTERFACE ${fmmio_SOURCE_DIR}/include)
endif()