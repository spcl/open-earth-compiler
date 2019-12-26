#ifndef MLIR_DIALECT_STENCIL_PASSES_H
#define MLIR_DIALECT_STENCIL_PASSES_H

#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stencil {

std::unique_ptr<OpPassBase<mlir::ModuleOp>>
createConvertStencilToStandardPass();

} // namespace stencil
} // namespace mlir

#endif // MLIR_DIALECT_STENCIL_PASSES_H
