set(LIB_LIBS
  MLIRAnalysis
  MLIRLLVMIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
)
add_llvm_library(OECOptLib
  oec-opt.cpp
)
target_link_libraries(OECOptLib ${LIB_LIBS})

set(LIBS
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRLoopsToGPU
  MLIRLinalgToLLVM

  MLIRLoopToStandard
  MLIREDSC
  MLIRFxpMathOps
  MLIRGPU
  MLIRGPUtoNVVMTransforms
  MLIRGPUtoROCDLTransforms
  MLIRGPUtoSPIRVTransforms
  MLIRLinalgOps
  MLIRLLVMIR
  MLIRLoopOps
  MLIRNVVMIR
  MLIROptMain
  MLIRParser
  MLIRPass
  MLIRQuantizerTransforms
  MLIRQuantOps
  MLIRROCDLIR
  MLIRSPIRV
  MLIRStandardToSPIRVTransforms
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
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
if(OEC_GPU_LOWERING_ENABLED)
  list(APPEND LIBS
    MLIRGPUtoCUDATransforms
  )
endif()
add_llvm_executable(oec-opt
 oec-opt.cpp
)

llvm_update_compile_flags(oec-opt)
whole_archive_link(oec-opt ${LIBS})
target_link_libraries(oec-opt PRIVATE MLIRIR OECOptLib ${LIBS} LLVMSupport)