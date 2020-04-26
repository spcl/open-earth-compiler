#ifndef CONVERSION_STENCILTOSTANDARD_PASSDETAIL_H_
#define CONVERSION_STENCILTOSTANDARD_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Conversion/StencilToStandard/Passes.h.inc"

} // end namespace mlir

#endif // CONVERSION_STENCILTOSTANDARD_PASSDETAIL_H_
