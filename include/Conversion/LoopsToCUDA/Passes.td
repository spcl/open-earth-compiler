#ifndef CONVERSION_LOOPSTOCUDA_PASSES
#define CONVERSION_LOOPSTOCUDA_PASSES

include "mlir/Pass/PassBase.td"

def LaunchFuncToRuntimeCallsPass : Pass<"stencil-gpu-to-runtime", "ModuleOp"> {
  let summary = "Convert all kernel launches to GPU runtime calls";
  let constructor = "mlir::createLaunchFuncToRuntimeCallsPass()";
  let options = [
    Option<"gpuBinaryAnnotation", "gpu-binary-annotation", "std::string",
           "\"nvvm.cubin\"",
           "Annotation attribute string for GPU binary">,
  ];
}

def StencilLoopMappingPass : FunctionPass<"stencil-loop-mapping"> {
  let summary = "Map parallel loops to blocks and threads";
  let constructor = "mlir::createStencilLoopMappingPass()";
  let options = [
    ListOption<"blockSizes", "block-sizes", "int64_t",
           "Block sizes used for the mapping",
           "llvm::cl::OneOrMore, llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

#endif // CONVERSION_LOOPSTOCUDA_PASSES
