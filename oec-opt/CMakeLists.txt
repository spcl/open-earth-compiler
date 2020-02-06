set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

set(LIB_LIBS
  MLIRAnalysis
  MLIRLLVMIR
  MLIROptMain
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
)
add_llvm_library(OECOptLib
  oec-opt.cpp
)
target_link_libraries(OECOptLib ${LIB_LIBS})

set(FULL_LINK_LIBS
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRDialect
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
  MLIRLinalgAnalysis
  MLIRLinalgEDSC
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRLLVMIR
  MLIRLoopOps
  MLIRNVVMIR
  MLIROpenMP
  MLIROptMain
  MLIRParser
  MLIRPass
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantizerTransforms
  MLIRQuantOps
  MLIRROCDLIR
  MLIRSPIRV
  MLIRLinalgToSPIRVTransforms
  MLIRStandardToSPIRVTransforms
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRTransformUtils
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
set(LIBS
  MLIRIR
  MLIROptMain
  LLVMSupport
  LLVMCore
  LLVMAsmParser
)

if(OEC_GPU_LOWERING_ENABLED)
  list(APPEND LIBS
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    MLIRTargetNVVMIR
    ${CUDA_RUNTIME_LIBRARY}
  )
  # Order matters here, so insert at front.
  list(INSERT FULL_LINK_LIBS 0
    MLIRGPUtoCUDATransforms
    OECGPUtoCUDATransforms
  )
endif()
add_llvm_executable(oec-opt
 oec-opt.cpp
)

llvm_update_compile_flags(oec-opt)
whole_archive_link(oec-opt ${FULL_LINK_LIBS})
target_link_libraries(oec-opt PRIVATE ${LIBS})