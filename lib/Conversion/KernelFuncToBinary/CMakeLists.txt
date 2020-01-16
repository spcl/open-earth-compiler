
if(BUILD_GPU_BACKEND)
  if (NOT ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building the mlir cuda runner requires the NVPTX backend")
  endif()
  llvm_map_components_to_libnames(nvptx "NVPTX")
  
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building the mlir cuda runner requires a working CUDA install")
  endif()
  # We need the libcuda.so library.
  find_library(CUDA_RUNTIME_LIBRARY cuda)

  add_llvm_library(MLIRKernelFuncToBinary
    ConvertKernelFuncToBinary.cpp
  )
  add_dependencies(
    MLIRKernelFuncToBinary

     MLIRGPU
    MLIRLLVMIR
    MLIRNVVMIR
    MLIRPass
    MLIRTargetNVVMIR
    ${nvptx}
  )
  target_link_libraries(
    MLIRKernelFuncToBinary
    MLIRGPU
    MLIRLLVMIR
    MLIRNVVMIR
    MLIRPass
    MLIRTargetNVVMIR
    ${nvptx}
    ${CUDA_RUNTIME_LIBRARY}
  )
  target_include_directories(
    MLIRKernelFuncToBinary
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()