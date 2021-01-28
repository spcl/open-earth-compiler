#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace stencil;

namespace {

// Introduce a peel loop if the shape is not a multiple of the unroll factor
struct PeelRewrite : public stencil::ApplyOpPattern {
  using ApplyOpPattern::ApplyOpPattern;

  void makePeelIteration(stencil::ReturnOp returnOp, unsigned tripCount,
                         PatternRewriter &rewriter) const {

    // Create empty store for all iterations that exceed the trip count
    SmallVector<Value, 16> newOperands;
    for (auto en : llvm::enumerate(returnOp.getOperands())) {
      if (en.index() % returnOp.getUnrollFac() >= tripCount) {
        auto resultOp = en.value().getDefiningOp();
        rewriter.updateRootInPlace(resultOp,
                                   [&]() { resultOp->setOperands({}); });
      }
    }
  }

  void addPeelIteration(stencil::ApplyOp applyOp, stencil::ReturnOp returnOp,
                        int64_t domainSize, PatternRewriter &rewriter) const {
    auto unrollFac = returnOp.getUnrollFac();
    auto unrollDim = returnOp.getUnrollDim();
    if (domainSize < unrollFac) {
      makePeelIteration(returnOp, domainSize, rewriter);
    } else {
      // Clone a peel and a body operation
      auto peelOp = cast<stencil::ApplyOp>(rewriter.clone(*applyOp));
      auto bodyOp = cast<stencil::ApplyOp>(rewriter.clone(*applyOp));

      // Adapt the shape of the two apply ops
      auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
      auto lb = shapeOp.getLB();
      auto ub = shapeOp.getUB();
      int64_t split = ub[unrollDim] - domainSize % unrollFac;
      lb[unrollDim] = split;
      ub[unrollDim] = split;

      // Introduce a second apply to handle the peel domain
      cast<ShapeOp>(peelOp.getOperation()).updateShape(lb, shapeOp.getUB());
      cast<ShapeOp>(bodyOp.getOperation()).updateShape(shapeOp.getLB(), ub);

      // Remove stores that exceed the domain
      makePeelIteration(
          cast<stencil::ReturnOp>(peelOp.getBody()->getTerminator()),
          domainSize % unrollFac, rewriter);

      // Introduce a stencil combine to replace the apply operation
      auto combineOp = rewriter.create<stencil::CombineOp>(
          applyOp.getLoc(), applyOp.getResultTypes(), unrollDim, split,
          bodyOp.getResults(), peelOp.getResults(), ValueRange(), ValueRange(),
          applyOp.lbAttr(), applyOp.ubAttr());
      rewriter.replaceOp(applyOp, combineOp.getResults());
    }
  }

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Get the return operation and the shape of the apply operation
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    auto shapeOp = cast<ShapeOp>(applyOp.getOperation());

    // Exit if there is no need for a peel loop
    auto unrollDim = returnOp.getUnrollDim();
    auto unrollFac = returnOp.getUnrollFac();
    auto domainSize = shapeOp.getUB()[unrollDim] - shapeOp.getLB()[unrollDim];
    if (domainSize % unrollFac == 0)
      return failure();

    // Exit if the apply op is inside the domain covered by the combine tree
    auto rootOp = applyOp.getCombineTreeRootShape();
    if (shapeOp.getLB()[unrollDim] != rootOp.getLB()[unrollDim])
      return failure();

    addPeelIteration(applyOp, returnOp, domainSize, rewriter);
    return success();
  }
};

struct PeelOddIterationsPass
    : public PeelOddIterationsPassBase<PeelOddIterationsPass> {
  void runOnFunction() override;
};

void PeelOddIterationsPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!StencilDialect::isStencilProgram(funcOp))
    return;

  // Check all combine op operands have one use
  auto result = funcOp.walk([&](stencil::CombineOp combineOp) {
    for (auto operand : combineOp.getOperands()) {
      if (!operand.hasOneUse())
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    funcOp.emitOpError("execute domain splitting before bounds unrolling");
    return signalPassFailure();
  }

  // Check shape inference has been executed
  result = funcOp->walk([&](stencil::ShapeOp shapeOp) {
    if (!shapeOp.hasShape())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    funcOp.emitOpError("execute shape inference before bounds unrolling");
    signalPassFailure();
    return;
  }

  // Populate the pattern list depending on the config
  OwningRewritePatternList patterns;
  patterns.insert<PeelRewrite>(&getContext());
  applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createPeelOddIterationsPass() {
  return std::make_unique<PeelOddIterationsPass>();
}
