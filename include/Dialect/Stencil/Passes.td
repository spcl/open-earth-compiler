#ifndef DIALECT_STENCIL_PASSES
#define DIALECT_STENCIL_PASSES

include "mlir/Pass/PassBase.td"

def CallInliningPass : Pass<"stencil-call-inlining", "ModuleOp"> {
  let summary = "Inline stencil functions";
  let constructor = "mlir::createParallelLoopFusionPass()";
}

def StencilInliningPass : FunctionPass<"stencil-inlining"> {
  let summary = "Inline stencil apply ops";
  let constructor = "mlir::createStencilInliningPass()";
}

def StencilUnrollingPass : FunctionPass<"stencil-unrolling"> {
  let summary = "Unroll stencil apply ops";
  let constructor = "mlir::createStencilUnrollingPass()";
  let options = [
    Option<"unrollFactor", "unroll-factor", "unsigned", /*default=*/"2",
           "Number of unrolled loop iterations">,
    Option<"unrollIndex", "unroll-index", "unsigned", /*default=*/"1",
           "Unroll index specifying the unrolling dimension">,
  ];
}

def ShapeInferencePass : FunctionPass<"stencil-shape-inference"> {
  let summary = "Infer loop bounds and storage shapes";
  let constructor = "mlir::createShapeInferencePass()";
}

def ShapeShiftPass : FunctionPass<"stencil-shape-shift"> {
  let summary = "Shift the loop bounds and offsets to the positive range";
  let constructor = "mlir::createShapeShiftPass()";
}

#endif // DIALECT_STENCIL_PASSES
