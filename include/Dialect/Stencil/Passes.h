#ifndef DIALECT_STENCIL_PASSES_H
#define DIALECT_STENCIL_PASSES_H


#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>> createStencilDomainSplitPass();

std::unique_ptr<OperationPass<FuncOp>> createStencilInliningPass();

std::unique_ptr<OperationPass<FuncOp>> createStencilUnrollingPass();

std::unique_ptr<OperationPass<FuncOp>> createStencilCombineLoweringPass();

std::unique_ptr<OperationPass<FuncOp>> createShapeInferencePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/Stencil/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_STENCIL_PASSES_H
