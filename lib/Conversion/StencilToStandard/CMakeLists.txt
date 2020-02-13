set(LIBS

  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRStencil
  MLIRLLVMIR
  MLIRLoopToStandard
  MLIRStandardToLLVM
  MLIRTransformUtils
  LLVMSupport)

add_llvm_library(MLIRStencilToStandard
  ConvertStencilToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/StencilToStandard
)
target_link_libraries(MLIRStencilToStandard ${LIBS})

# specify header directories
target_include_directories(MLIRStencilToStandard PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(MLIRStencilToStandard PUBLIC "${PROJECT_BINARY_DIR}/include")