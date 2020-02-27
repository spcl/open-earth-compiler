#ifndef MLIR_CONVERSION_KERNELTOCUDA_PASSES_H
#define MLIR_CONVERSION_KERNELTOCUDA_PASSES_H

#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stencil {

std::unique_ptr<OpPassBase<ModuleOp>> createLaunchFuncToCUDACallsPass();

std::unique_ptr<OpPassBase<LLVM::LLVMFuncOp>> createIndexOptimizationPass();

std::unique_ptr<OpPassBase<FuncOp>> createStencilLoopMappingPass();

OwnedCubin compilePtxToCubin(const std::string &ptx, Location loc,
                             StringRef name);

} // namespace stencil
} // namespace mlir

#endif // MLIR_CONVERSION_KERNELTOCUDA_PASSES_H
