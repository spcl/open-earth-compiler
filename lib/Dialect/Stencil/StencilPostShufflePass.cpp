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

// TODO factor this method to a separate file

// Helper method skipping unary operations
Operation *skipUnaryOperations(Operation *op) {
  while (auto negOp = dyn_cast_or_null<NegFOp>(op)) {
    op = op->getOperand(0).getDefiningOp();
  }
  return op;
}

// Helper method that returns true if first argument is produced before
bool isProducedBeforeOrSame(Value before, Value after) {
  auto beforeOp = skipUnaryOperations(before.getDefiningOp());
  auto afterOp = skipUnaryOperations(after.getDefiningOp());
  if (beforeOp && afterOp) {
    return beforeOp == afterOp || beforeOp->isBeforeInBlock(afterOp);
  }
  return false;
}

#include "Dialect/Stencil/StencilPostShufflePatterns.cpp.inc"

struct StencilPostShufflePass : public OperationPass<StencilPostShufflePass, stencil::ApplyOp> {
  void runOnOperation() override;
};

void StencilPostShufflePass::runOnOperation() {
  auto applyOp = getOperation();

  OwningRewritePatternList patterns;
  populateWithGenerated(&getContext(), &patterns);
  applyPatternsGreedily(applyOp, patterns);
}

} // namespace

std::unique_ptr<OpPassBase<stencil::ApplyOp>>
stencil::createStencilPostShufflePass() {
  return std::make_unique<StencilPostShufflePass>();
}
