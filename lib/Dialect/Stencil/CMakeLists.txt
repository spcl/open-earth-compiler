
add_llvm_library(MLIRStencil
  DialectRegistration.cpp
  StencilDialect.cpp
  StencilOps.cpp
  StencilTypes.cpp
  CallInliningPass.cpp
  StencilInliningPass.cpp
  ShapeInferencePass.cpp
  ShapeShiftPass.cpp
)

# Make sure that the TableGen generated files are up-to-date
add_dependencies(MLIRStencil
  MLIRStencilOpsIncGen)

target_link_libraries(MLIRStencil
  MLIRIR
  MLIRParser
  MLIRSupport)

# specify header directories
target_include_directories(MLIRStencil PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(MLIRStencil PUBLIC "${PROJECT_BINARY_DIR}/include")