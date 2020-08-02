add_llvm_library(MLIRTargetStencil
  Stencil/RandomStencilGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/Stencil
  DEPENDS
  intrinsics_gen
  )
target_link_libraries(MLIRTargetStencil
  Stencil
  MLIRIR
  )
