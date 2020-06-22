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

#endif // CONVERSION_LOOPSTOCUDA_PASSES
