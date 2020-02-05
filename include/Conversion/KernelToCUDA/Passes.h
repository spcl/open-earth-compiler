#ifndef MLIR_CONVERSION_KERNELTOCUDA_PASSES_H
#define MLIR_CONVERSION_KERNELTOCUDA_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<OpPassBase<LLVM::LLVMFuncOp>>
createIndexOptimizationPass();

} // namespace mlir

#endif // MLIR_CONVERSION_KERNELTOCUDA_PASSES_H
