set(LIBS
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRLinalgToLLVM

  MLIRLoopToStandard
  MLIREDSC
  MLIRFxpMathOps
  MLIRLinalg
  MLIRLLVMIR
  MLIRLoopOps
  MLIROptMain
  MLIRParser
  MLIRPass
  MLIRQuantizerTransforms
  MLIRQuantOps
  #MLIRSPIRV
  #MLIRStandardToSPIRVTransforms
  #MLIRSPIRVTestPasses
  #MLIRSPIRVTransforms
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRVectorOps
  MLIRVectorToLLVM
  MLIRVectorToLoops
  
  MLIRStencil
  MLIRStencilToStandard
)
if(BUILD_GPU_BACKEND)
  list(APPEND LIBS
    MLIRLoopsToGPU
    MLIRGPU
    MLIRGPUtoNVVMTransforms
    MLIRNVVMIR
    MLIRKernelFuncToBinary) 
endif()

add_llvm_executable(oec-opt
 oec-opt.cpp
)
llvm_update_compile_flags(oec-opt)
whole_archive_link(oec-opt ${LIBS})
target_link_libraries(oec-opt PRIVATE MLIRIR MLIRMlirOptLib ${LIBS} LLVMSupport)