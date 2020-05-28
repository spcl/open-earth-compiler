set(NVPTX_LIBS
  MC
  NVPTXCodeGen
  NVPTXDesc
  NVPTXInfo
)
set(AMDGPU_LIBS
  MC
  AllTargetsAsmParsers
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
)

add_mlir_dialect_library(GPUtoCUDATransforms
  ConvertLaunchFuncToCUDACalls.cpp
  ConvertKernelFuncToCubin.cpp
  StencilIndexOptimizationPass.cpp
  StencilLoopMappingPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/LoopsToCUDA

  LINK_COMPONENTS
  ${NVPTX_LIBS}
  ${AMDGPU_LIBS}

  LINK_LIBS PUBLIC
  MLIRROCDLIR
  MLIRTargetROCDLIR
  

  DEPENDS
  MLIRLoopsToCUDAPassIncGen
)

target_link_libraries(GPUtoCUDATransforms PUBLIC MLIRIR)