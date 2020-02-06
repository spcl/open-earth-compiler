
if(OEC_GPU_LOWERING_ENABLED)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building the GPU lowering of oec-opt requires CUDA")
  endif()
  find_library(CUDA_RUNTIME_LIBRARY cuda)

  set(LIBS

    MLIRIR
    MLIRStandardOps
    )

  add_llvm_library(OECGPUtoCUDATransforms
    ConvertLaunchFuncToCUDACalls.cpp
    ConvertKernelFuncToCubin.cpp
    IndexOptimizationPass.cpp
  )

  add_dependencies(OECGPUtoCUDATransforms ${LIBS})
  target_link_libraries(OECGPUtoCUDATransforms ${LIBS})

  target_include_directories(OECGPUtoCUDATransforms PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(OECGPUtoCUDATransforms PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(OECGPUtoCUDATransforms PUBLIC "${PROJECT_BINARY_DIR}/include")
endif()