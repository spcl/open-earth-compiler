set(LIBS
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  #MLIRLoopsToGPU
  #MLIRLinalgToLLVM

  MLIRLoopToStandard
  MLIREDSC
  MLIRFxpMathOps
  #MLIRGPU
  #MLIRGPUtoNVVMTransforms
  #MLIRGPUtoROCDLTransforms
  #MLIRGPUtoSPIRVTransforms
  #MLIRLinalg
  MLIRLLVMIR
  MLIRLoopOps
  #MLIRNVVMIR
  MLIROptMain
  MLIRParser
  MLIRPass
  MLIRQuantizerTransforms
  MLIRQuantOps
  #MLIRROCDLIR
  #MLIRSPIRV
  #MLIRStandardToSPIRVTransforms
  #MLIRSPIRVTransforms
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransforms
  #MLIRTestDialect
  #MLIRTestIR
  #MLIRTestPass
  #MLIRTestTransforms
  MLIRSupport
  MLIRVectorOps
  MLIRVectorToLLVM
  MLIRVectorToLoops
  
  MLIRStencil
  #MLIRStencilToStandard
)
#if(MLIR_CUDA_CONVERSIONS_ENABLED)
#  list(APPEND LIBS
#    MLIRGPUtoCUDATransforms
#  )
#endif()

add_llvm_executable(oec-opt
 oec-opt.cpp
)
llvm_update_compile_flags(oec-opt)
whole_archive_link(oec-opt ${LIBS})
target_link_libraries(oec-opt PRIVATE MLIRIR MLIRMlirOptLib ${LIBS} LLVMSupport)
target_link_directories(oec-opt PRIVATE ${PROJECT_BINARY_DIR}/lib/Dialect/Stencil)
target_link_directories(oec-opt PRIVATE ${PROJECT_BINARY_DIR}/lib/Conversion/StencilToStandard)
