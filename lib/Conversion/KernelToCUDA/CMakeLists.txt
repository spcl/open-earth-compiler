
if(OEC_GPU_LOWERING_ENABLED)
  if (NOT ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building the mlir cuda runner requires the NVPTX backend")
  endif()
  llvm_map_components_to_libnames(nvptx "NVPTX")
  

  add_llvm_library(MLIRKernelToCUDA
    ConvertKernelFuncToPtx.cpp
    ConvertLaunchFuncToCUDA.cpp
  )
  add_dependencies(
    MLIRKernelToCUDA

    MLIRGPU
    MLIRLLVMIR
    MLIRNVVMIR
    MLIRPass
    MLIRTargetNVVMIR
    ${nvptx}
  )
  target_link_libraries(
    MLIRKernelToCUDA
    MLIRGPU
    MLIRLLVMIR
    MLIRNVVMIR
    MLIRPass
    MLIRTargetNVVMIR
    ${nvptx}
  )
endif()