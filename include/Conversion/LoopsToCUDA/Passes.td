#ifndef CONVERSION_LOOPSTOCUDA_PASSES
#define CONVERSION_LOOPSTOCUDA_PASSES

include "mlir/Pass/PassBase.td"

def LaunchFuncToCUDACallsPass : Pass<"stencil-gpu-to-cuda", "ModuleOp"> {
  let summary = "Convert all kernel launches to CUDA runtime calls";
  let constructor = "mlir::createLaunchFuncToCUDACallsPass()";
}

def StencilIndexOptimizationPass : Pass<"stencil-index-optimization", "LLVM::LLVMFuncOp"> {
  let summary = "Convert 64-bit index computations to 32-bit index computations";
  let constructor = "mlir::createStencilIndexOptimizationPass()";
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
