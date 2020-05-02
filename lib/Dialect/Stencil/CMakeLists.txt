add_mlir_dialect_library(Stencil
  StencilDialect.cpp
  StencilOps.cpp
  StencilTypes.cpp
  StencilInliningPass.cpp
  ShapeInferencePass.cpp
  ShapeShiftPass.cpp
  StencilUnrollingPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Stencil

  DEPENDS
  MLIRStencilOpsIncGen
  MLIRStencilPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

