if(OEC_CUDA_BACKEND_ENABLED)
  set(NVPTX_LIB
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
  set(NVVMIR_LIB
    MLIRNVVMIR
    MLIRTargetNVVMIR
  )
endif()

add_mlir_dialect_library(GPUToKernelAndRuntimeCalls
  ConvertLaunchFuncToCUDACalls.cpp
  ConvertKernelFuncToCubin.cpp
  StencilLoopMappingPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/LoopsToCUDA

  LINK_COMPONENTS
  Core
  MC
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  ${NVVMIR_LIB}

  DEPENDS
  MLIRLoopsToCUDAPassIncGen
)

target_link_libraries(GPUToKernelAndRuntimeCalls PUBLIC MLIRIR)