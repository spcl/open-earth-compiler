set(LIBS
  MLIRAffineOps
  # MLIRLoopsToGPU
  MLIRAnalysis
  MLIRLoopToStandard
  MLIREDSC
  MLIRFxpMathOps
  MLIRGPU
  MLIRGPUtoNVVMTransforms
  MLIRGPUtoROCDLTransforms
  MLIRGPUtoSPIRVTransforms
  MLIRLinalg
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
  # MLIRSPIRVConversion
  MLIRSPIRVTransforms
  MLIRStandardOps
  MLIRStandardToLLVM
  # MLIRStencil
  # MLIRStencilIIR
  # MLIRStencilConversion
  # MLIRStencilToAffine
  MLIRTransforms
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  # MLIRTestTransforms
  MLIRSupport
  MLIRVectorOps
  # MLIRVectorToLLVM
  # MLIRSten
)
if(MLIR_CUDA_CONVERSIONS_ENABLED)
  list(APPEND LIBS
    MLIRGPUtoCUDATransforms
  )
endif()

add_executable(occ-opt
 mlir-opt.cpp
)
# llvm_update_compile_flags(mlir-opt)
target_compile_options(occ-opt PRIVATE -fno-rtti)
whole_archive_link(occ-opt ${LIBS})
target_link_libraries(occ-opt PRIVATE MLIRIR MLIRMlirOptLib ${LIBS} LLVMSupport)
