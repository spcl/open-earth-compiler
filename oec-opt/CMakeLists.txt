set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

set(LIBS
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
  MLIRGPUtoCUDATransforms
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
  MLIROptLib
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
  MLIRSupport
  MLIRVectorOps
  MLIRVectorToLLVM
  MLIRVectorToLoops
  MLIRIR
  LLVMSupport
  LLVMCore
  LLVMAsmParser

  MLIRStencil
  MLIRStencilToStandard
  OECGPUtoCUDATransforms
  ${CUDA_RUNTIME_LIBRARY}
)
add_llvm_executable(oec-opt
 oec-opt.cpp
)

# Manually expand the target library, since our MLIR libraries
# aren't plugged into the LLVM dependency tracking. If we don't
# do this then we can't insert the CodeGen library after ourselves
llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
# Prepend LLVM in front of every target, this is how the library
# are named with CMake
SET(targets_to_link)
FOREACH(t ${TARGET_LIBS})
  LIST(APPEND targets_to_link "LLVM${t}")
ENDFOREACH(t)

llvm_update_compile_flags(oec-opt)
target_link_libraries(oec-opt PRIVATE ${LIBS} ${targets_to_link})