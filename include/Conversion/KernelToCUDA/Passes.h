#ifndef MLIR_CONVERSION_KERNELTOCUDA_PASSES_H
#define MLIR_CONVERSION_KERNELTOCUDA_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stencil {

std::unique_ptr<OpPassBase<ModuleOp>> createLaunchFuncToCUDACallsPass();

std::unique_ptr<OpPassBase<LLVM::LLVMFuncOp>> createIndexOptimizationPass();

void createGPUToCubinPipeline(OpPassManager &pm);

} // namespace stencil
} // namespace mlir

#endif // MLIR_CONVERSION_KERNELTOCUDA_PASSES_H
