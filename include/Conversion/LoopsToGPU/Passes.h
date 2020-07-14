#ifndef CONVERSION_LOOPSTOGPU_PASSES_H
#define CONVERSION_LOOPSTOGPU_PASSES_H

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {

class Pass;

void registerGPUToCUBINPipeline();
void registerGPUToHSACOPipeline();

} // namespace mlir

#endif // CONVERSION_LOOPSTOGPU_PASSES_H
