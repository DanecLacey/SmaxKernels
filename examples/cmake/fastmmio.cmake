CPMAddPackage(
  NAME fmmio
  GITHUB_REPOSITORY alugowski/fast_matrix_market
  GIT_TAG v1.7.6
)

if(fmmio_ADDED)
  # 1) Build the implementation file into a real archive
  add_library(fmmio STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../mmio.cpp
  )

  # 2) Tell users of fmmio where to find the FAST_MMIO headers
  target_include_directories(fmmio
    PUBLIC
      ${fmmio_SOURCE_DIR}/include      # the external header-only lib
      ${CMAKE_CURRENT_LIST_DIR}/..      # for your local mmio.cpp
  )

  # 3) Expose the FAST_MMIO compile‑definition 
  target_compile_definitions(fmmio
    PUBLIC USE_FAST_MMIO
  )

endif()