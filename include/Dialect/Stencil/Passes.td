#ifndef DIALECT_STENCIL_PASSES
#define DIALECT_STENCIL_PASSES

include "mlir/Pass/PassBase.td"

def DomainSplitPass : FunctionPass<"stencil-domain-split"> {
  let summary = "Copy apply ops used in multiple domains";
  let constructor = "mlir::createDomainSplitPass()";
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

def CombineToIfElsePass : FunctionPass<"stencil-combine-to-ifelse"> {
  let summary = "Lower stencil combine ops to if/else";
  let constructor = "mlir::createCombineToIfElsePass()";
  let options = [
    Option<"prepareOnly", "prepare-only", "bool", /*default=*/"false", 
           "Lower extra operands and fuse multiple producers attached to one combine">,
    Option<"internalOnly", "internal-only", "bool", /*default=*/"false", 
           "Lower only combine ops embedded in between apply ops">,
  ];
}

def ShapeInferencePass : FunctionPass<"stencil-shape-inference"> {
  let summary = "Infer loop bounds and storage shapes";
  let constructor = "mlir::createShapeInferencePass()";
  let options = [
    Option<"extendStorage", "extend-storage", "bool", /*default=*/"false", 
           "Extend the shape of store and buffer ops">,
  ];
}

def ShapeOverlapPass : FunctionPass<"stencil-shape-overlap"> {
  let summary = "Handle overlapping stores";
  let constructor = "mlir::createShapeOverlapPass()";
}

def StorageMaterializationPass : FunctionPass<"stencil-storage-materialization"> {
  let summary = "Introduce explicit storage between combine and apply ops";
  let constructor = "mlir::createStorageMaterializationPass()";
}

def PeelOddIterationsPass : FunctionPass<"stencil-peel-odd-iterations"> {
  let summary = "Peel odd iterations that are not a multiple of the unroll factor";
  let constructor = "mlir::createPeelOddIterationsPass()";
}

#endif // DIALECT_STENCIL_PASSES
