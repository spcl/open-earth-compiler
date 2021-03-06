if(CUDA_BACKEND_ENABLED)
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
if(ROCM_BACKEND_ENABLED)
  set(AMDGPU_LIBS
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
  set(ROCDLIR_LIB
    lldCommon
    lldDriver
    lldELF
    MLIRROCDLIR
    MLIRTargetROCDLIR
  )
endif()

add_mlir_dialect_library(GPUToKernelAndRuntimeCalls
  ConvertKernelFuncToCubin.cpp
  ConvertKernelFuncToHsaco.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/LoopsToGPU
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}

  LINK_COMPONENTS
  Core
  MC
  MCParser
  ${NVPTX_LIBS}
  ${AMDGPU_LIBS}
  
  LINK_LIBS PUBLIC
  ${NVVMIR_LIB}
  ${ROCDLIR_LIB}
)
target_link_libraries(GPUToKernelAndRuntimeCalls PUBLIC MLIRIR)