set(LIBS
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport)

add_llvm_library(MLIRStencil
  StencilDialect.cpp
  StencilOps.cpp
  StencilTypes.cpp
  CallInliningPass.cpp
  StencilInliningPass.cpp
  StencilRegPass.cpp
  ShapeInferencePass.cpp
  ShapeShiftPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Stencil
)

# Make sure that the TableGen generated files are up-to-date
add_dependencies(MLIRStencil
  MLIRStencilOpsIncGen
  MLIRStencilRegPassPatternsIncGen
  ${LIBS})
target_link_libraries(MLIRStencil ${LIBS})

# specify header directories
target_include_directories(MLIRStencil PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(MLIRStencil PUBLIC "${PROJECT_BINARY_DIR}/include")

