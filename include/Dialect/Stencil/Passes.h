#ifndef MLIR_DIALECT_STENCIL_PASSES_H
#define MLIR_DIALECT_STENCIL_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stencil {

std::unique_ptr<OpPassBase<ModuleOp>> createCallInliningPass();

std::unique_ptr<OpPassBase<FuncOp>> createStencilInliningPass();

std::unique_ptr<OpPassBase<FuncOp>> createShapeShiftPass();

std::unique_ptr<OpPassBase<FuncOp>> createShapeInferencePass();

std::unique_ptr<OpPassBase<FuncOp>> createStencilUnrollingPass();

} // namespace stencil
} // namespace mlir

#endif // MLIR_DIALECT_STENCIL_PASSES_H
