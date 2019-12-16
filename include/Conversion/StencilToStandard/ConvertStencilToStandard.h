#ifndef MLIR_CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H
#define MLIR_CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H

#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Dialect/Stencil/StencilOps.h"

namespace mlir {

void populateStencilToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx);
}

#endif // MLIR_CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H
