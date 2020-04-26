#ifndef DIALECT_STENCIL_PASSDETAIL_H_
#define DIALECT_STENCIL_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Dialect/Stencil/Passes.h.inc"

} // end namespace mlir

#endif // DIALECT_STENCIL_PASSDETAIL_H_
