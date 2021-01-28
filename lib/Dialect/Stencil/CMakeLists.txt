add_mlir_dialect_library(Stencil
  StencilUtils.cpp
  StencilDialect.cpp
  StencilOps.cpp
  StencilTypes.cpp
  StencilInliningPass.cpp
  ShapeInferencePass.cpp
  ShapeOverlapPass.cpp
  StencilUnrollingPass.cpp
  CombineToIfElsePass.cpp
  DomainSplitPass.cpp
  StorageMaterializationPass.cpp
  PeelOddIterationsPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Stencil

  DEPENDS
  MLIRStencilOpsIncGen
  MLIRStencilPassIncGen
  MLIRStencilInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

