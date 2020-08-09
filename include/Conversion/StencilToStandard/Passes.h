#ifndef CONVERSION_STENCILTOSTANDARD_PASSES_H
#define CONVERSION_STENCILTOSTANDARD_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createConvertStencilToStandardPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/StencilToStandard/Passes.h.inc"

} // namespace mlir

#endif // CONVERSION_STENCILTOSTANDARD_PASSES_H
