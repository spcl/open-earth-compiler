#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTraits.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>

using namespace mlir;

namespace {

// Helper method computing the minimal access offset per field
llvm::DenseMap<Value, int64_t> computeMinJOffset(Value value) {
  llvm::DenseMap<Value, int64_t> result;
  // Return an empty map for arguments
  auto op = value.getDefiningOp();
  if (!op) 
    return result;
  // Return the current map offset
  if (auto accessOp = dyn_cast<stencil::AccessOp>(op)) {
    if(llvm::is_contained(accessOp.getViewType().getDimensions(), stencil::kUnrollDimension))
      result[accessOp.getOperand()] = accessOp.getOffset()[stencil::kUnrollDimension];
    return result;
  }
  // Compute the offset recursively
  for (auto operand : op->getOperands()) {
    auto offsets = computeMinJOffset(operand);
    for (auto offset : offsets) {
      if (result.count(offset.getFirst()) == 0)
        result[offset.getFirst()] = offset.getSecond();
      else
        result[offset.getFirst()] =
            std::min(offset.getSecond(), result[offset.getFirst()]);
    }
  }
  return result;
}

// Helper method that orders values according to the accessed j-offsets
bool isAccessedBefore(Value before, Value after) {
  llvm::DenseMap<Value, int64_t> beforeOffsets = computeMinJOffset(before);
  llvm::DenseMap<Value, int64_t> afterOffsets = computeMinJOffset(after);
  int64_t beforeWins = 0;
  int64_t afterWins = 0;
  for(auto beforeOffset : beforeOffsets) {
    if(afterOffsets.count(beforeOffset.getFirst()) != 0) {
      if(beforeOffset.getSecond() < afterOffsets[beforeOffset.getFirst()])
        beforeWins++;
      else 
        afterWins++;
    }
  }
  return beforeWins > afterWins;
}

// Helper method that returns true if first argument is produced before
// bool isProducedBefore(Value before, Value after) {
//   auto beforeOp = before.getDefiningOp();
//   auto afterOp = after.getDefiningOp();
//   if (beforeOp && afterOp) {
//     return beforeOp->isBeforeInBlock(afterOp);
//   }
//   return before == nullptr;
// }

// Helper method to check if a value was produced by a specific operation type
template <typename TOp>
bool isProducedBy(Value value) {
  if (auto definingOp = value.getDefiningOp())
    return isa<TOp>(definingOp);
  return false;
}

#include "Dialect/Stencil/StencilShufflePatterns.cpp.inc"

struct StencilShufflePass
    : public OperationPass<StencilShufflePass, stencil::ApplyOp> {
  void runOnOperation() override;
};

void StencilShufflePass::runOnOperation() {
  auto applyOp = getOperation();

  OwningRewritePatternList patterns;
  populateWithGenerated(&getContext(), &patterns);
  applyPatternsGreedily(applyOp, patterns);
}

} // namespace

std::unique_ptr<OpPassBase<stencil::ApplyOp>>
stencil::createStencilShufflePass() {
  return std::make_unique<StencilShufflePass>();
}
