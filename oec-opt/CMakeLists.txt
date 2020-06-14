get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRTestTransforms
  
  Stencil
  StencilToStandard
  GPUToKernelAndRuntimeCalls
  ${CUDA_RUNTIME_LIBRARY}
)

add_llvm_executable(oec-opt oec-opt.cpp)

llvm_update_compile_flags(oec-opt)
target_link_libraries(oec-opt PRIVATE ${LIBS})

