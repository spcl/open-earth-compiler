#ifndef CONVERSION_LOOPSTOCUDA_PASSES_H
#define CONVERSION_LOOPSTOCUDA_PASSES_H

#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {

class Pass;

std::unique_ptr<Pass> createLaunchFuncToCUDACallsPass();

std::unique_ptr<Pass> createStencilIndexOptimizationPass();

std::unique_ptr<OperationPass<FuncOp>> createStencilLoopMappingPass();

OwnedCubin compilePtxToCubin(const std::string &ptx, Location loc,
                             StringRef name);

void registerGPUToCUBINPipeline();

} // namespace mlir

#endif // CONVERSION_LOOPSTOCUDA_PASSES_H
