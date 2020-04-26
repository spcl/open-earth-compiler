add_mlir_dialect_library(StencilToStandard
  ConvertStencilToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/StencilToStandard

  DEPENDS
  MLIRStencilToStandardPassIncGen
)

target_link_libraries(StencilToStandard PUBLIC MLIRIR)