#ifndef DIALECT_STENCIL_PASSES
#define DIALECT_STENCIL_PASSES

include "mlir/Pass/PassBase.td"

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

def StencilCombineLoweringPass : FunctionPass<"stencil-combine-lowering"> {
  let summary = "Lower stencil combines to if/else";
  let constructor = "mlir::createStencilCombineLoweringPass()";
}

def ShapeInferencePass : FunctionPass<"stencil-shape-inference"> {
  let summary = "Infer loop bounds and storage shapes";
  let constructor = "mlir::createShapeInferencePass()";
}

#endif // DIALECT_STENCIL_PASSES
