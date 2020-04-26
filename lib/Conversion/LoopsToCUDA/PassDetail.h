#ifndef CONVERSION_LOOPSTOCUDA_PASSDETAIL_H_
#define CONVERSION_LOOPSTOCUDA_PASSDETAIL_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Conversion/LoopsToCUDA/Passes.h.inc"

} // end namespace mlir

#endif // CONVERSION_LOOPSTOCUDA_PASSDETAIL_H_
