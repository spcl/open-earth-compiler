#ifndef CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD
#define CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD

include "mlir/Pass/PassBase.td"

def StencilToStandardPass : Pass<"convert-stencil-to-standard", "ModuleOp"> {
  let summary = "Convert stencil dialect to standard operations";
  let constructor = "mlir::createConvertStencilToStandardPass()";
}

#endif // CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD
