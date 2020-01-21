
if(OEC_GPU_LOWERING_ENABLED)
  if (NOT ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR "Building the GPU lowering of oec-opt requires NVPTX")
  endif()
  llvm_map_components_to_libnames(nvptx "NVPTX")

  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building the GPU lowering of oec-opt requires CUDA")
  endif()
  find_library(CUDA_RUNTIME_LIBRARY cuda)

  add_llvm_library(OECGPUtoCUDATransforms
    ConvertLaunchFuncToCUDACalls.cpp
    ConvertKernelFuncToCubin.cpp
  )
  target_include_directories(
    OECGPUtoCUDATransforms
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(
    OECGPUtoCUDATransforms

    MLIRGPU
    MLIRLLVMIR
    MLIRNVVMIR
    MLIRPass
    MLIRTargetNVVMIR
    ${nvptx}
    ${CUDA_RUNTIME_LIBRARY}
  )
endif()