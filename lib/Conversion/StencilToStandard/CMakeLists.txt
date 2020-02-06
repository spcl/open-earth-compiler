set(LIBS

  MLIRIR
  MLIRStandardOps
  )

add_llvm_library(MLIRStencilToStandard
  ConvertStencilToStandard.cpp
)
target_link_libraries(MLIRStencilToStandard ${LIBS})

# specify header directories
target_include_directories(MLIRStencilToStandard PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(MLIRStencilToStandard PUBLIC "${PROJECT_BINARY_DIR}/include")