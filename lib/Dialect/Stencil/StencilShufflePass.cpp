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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>
#include <limits>
#include <tuple>
#include <utility>

using namespace mlir;

namespace {

// Helper method computing the minimal and maximal access offset per field
llvm::DenseMap<Value, std::tuple<int64_t, int64_t>>
computeOffsetRange(Value value) {
  llvm::DenseMap<Value, std::tuple<int64_t, int64_t>> result;
  // Return an empty map for arguments
  auto op = value.getDefiningOp();
  if (!op)
    return result;
  // Return the current map offset
  if (auto accessOp = dyn_cast<stencil::AccessOp>(op)) {
    if (llvm::is_contained(accessOp.getViewType().getDimensions(),
                           1))
      result[accessOp.getOperand()] =
          std::make_tuple(accessOp.getOffset()[1],
                          accessOp.getOffset()[1]);
    return result;
  }
  // Compute the offset recursively
  for (auto operand : op->getOperands()) {
    auto offsets = computeOffsetRange(operand);
    for (auto offset : offsets) {
      if (result.count(offset.getFirst()) == 0)
        result[offset.getFirst()] = offset.getSecond();
      else
        result[offset.getFirst()] =
            std::make_tuple(std::min(std::get<0>(offset.getSecond()),
                                     std::get<0>(result[offset.getFirst()])),
                            std::max(std::get<1>(offset.getSecond()),
                                     std::get<1>(result[offset.getFirst()])));
    }
  }
  return result;
}

// Helper method that orders values according to the accessed j-offsets
bool isAccessedBefore(Value before, Value after) {
  // llvm::errs() << "before " << before.getDefiningOp() << " after " << after.getDefiningOp() << "\n";

  llvm::DenseMap<Value, std::tuple<int64_t, int64_t>> beforeOffsets =
      computeOffsetRange(before);
  llvm::DenseMap<Value, std::tuple<int64_t, int64_t>> afterOffsets =
      computeOffsetRange(after);

  int64_t beforeMin = std::numeric_limits<int64_t>::max();
  for(auto range : beforeOffsets) {
    beforeMin = std::min(beforeMin, std::get<0>(range.getSecond()));
  }

  int64_t afterMin = std::numeric_limits<int64_t>::max();
  for(auto range : afterOffsets) {
    afterMin = std::min(afterMin, std::get<0>(range.getSecond()));
  }
  
  return beforeMin < afterMin;

  // int64_t beforeWins = 0;
  // int64_t afterWins = 0;
  // for (auto beforeOffset : beforeOffsets) {
  //   if (afterOffsets.count(beforeOffset.getFirst()) != 0) {
  //     if (std::get<1>(beforeOffset.getSecond()) <=
  //         std::get<0>(afterOffsets[beforeOffset.getFirst()]))
  //       beforeWins++;
  //     if (std::get<0>(beforeOffset.getSecond()) >
  //         std::get<1>(afterOffsets[beforeOffset.getFirst()]))
  //       afterWins++;
  //   }

  //   llvm::errs() << "before min" << std::get<0>(beforeOffset.getSecond()) << " max " << std::get<1>(beforeOffset.getSecond()) << "\n";
  //   llvm::errs() << "after min" << std::get<0>(afterOffsets[beforeOffset.getFirst()]) << " max " << std::get<1>(afterOffsets[beforeOffset.getFirst()]) << "\n";
    
  // }

  // // TODO debug
  // llvm::errs() << "before " << beforeWins << " after " << afterWins << "\n";
  // return beforeWins > afterWins;
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