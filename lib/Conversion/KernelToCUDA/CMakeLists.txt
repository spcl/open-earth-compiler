
if(OEC_GPU_LOWERING_ENABLED)
  set(LIBS

    MLIRIR
    MLIRStandardOps
    )

  add_llvm_library(OECGPUtoCUDATransforms
    ConvertLaunchFuncToCUDACalls.cpp
    ConvertKernelFuncToCubin.cpp
    IndexOptimizationPass.cpp
  )

  target_link_libraries(OECGPUtoCUDATransforms ${LIBS})

  target_include_directories(OECGPUtoCUDATransforms PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(OECGPUtoCUDATransforms PUBLIC "${PROJECT_BINARY_DIR}/include")
endif()