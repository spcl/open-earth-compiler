set(NVPTX_LIB
  NVPTXCodeGen
  NVPTXDesc
  NVPTXInfo
)
#set(AMDGPU_LIBS
#  AMDGPUCodeGen
#  AMDGPUDesc
#  AMDGPUInfo
#)

add_mlir_dialect_library(GPUtoCUDATransforms
  ConvertLaunchFuncToCUDACalls.cpp
  ConvertKernelFuncToCubin.cpp
  StencilLoopMappingPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/LoopsToCUDA

  LINK_COMPONENTS
  Core
  MC
  ${NVPTX_LIBS}
  #${AMDGPU_LIBS}

  LINK_LIBS PUBLIC
  MLIRNVVMIR
  MLIRTargetNVVMIR
  #MLIRROCDLIR
  #MLIRTargetROCDLIR

  DEPENDS
  MLIRLoopsToCUDAPassIncGen
)

target_link_libraries(GPUtoCUDATransforms PUBLIC MLIRIR)