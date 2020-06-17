#ifndef CONVERSION_LOOPSTOCUDA_PASSES_H
#define CONVERSION_LOOPSTOCUDA_PASSES_H

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {

class Pass;

std::unique_ptr<Pass>
createLaunchFuncToRuntimeCallsPass(StringRef gpuBinaryAnnotation = "");

void registerGPUToCUBINPipeline();
void registerGPUToHSACOPipeline();

} // namespace mlir

#endif // CONVERSION_LOOPSTOCUDA_PASSES_H
