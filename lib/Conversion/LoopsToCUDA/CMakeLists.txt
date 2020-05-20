add_mlir_dialect_library(GPUtoCUDATransforms
  ConvertLaunchFuncToCUDACalls.cpp
  ConvertKernelFuncToCubin.cpp
  StencilLoopMappingPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/LoopsToCUDA

  DEPENDS
  MLIRLoopsToCUDAPassIncGen
)

target_link_libraries(GPUtoCUDATransforms PUBLIC MLIRIR)