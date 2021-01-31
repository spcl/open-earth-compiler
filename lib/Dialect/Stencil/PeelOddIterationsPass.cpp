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
#include "mlir/IR/Visitors.h"
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

  LogicalResult makePeelIteration(stencil::ApplyOp applyOp, int64_t peelSize,
                                  PatternRewriter &rewriter) const {
    // Get shape and terminator of the apply operation
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    auto shapeOp = cast<ShapeOp>(applyOp.getOperation());

    // Compute count of left or right empty stores
    auto leftCount = peelSize < 0 ? -peelSize : 0;
    auto rightCount = peelSize > 0 ? returnOp.getUnrollFac() - peelSize
                                   : returnOp.getUnrollFac();
    // Create empty store for all iterations that exceed the trip count
    unsigned numOfOperands = 0;
    SmallVector<Value, 16> newOperands;
    for (auto en : llvm::enumerate(returnOp.getOperands())) {
      int64_t unrollIdx = en.index() % returnOp.getUnrollFac();
      if (unrollIdx < leftCount || unrollIdx >= rightCount) {
        auto resultOp = en.value().getDefiningOp();
        numOfOperands += resultOp->getNumOperands();
        rewriter.updateRootInPlace(resultOp,
                                   [&]() { resultOp->setOperands({}); });
      }
    }

    // Extend the shape for negative peel sizes
    if (peelSize < 0) {
      auto lb = shapeOp.getLB();
      lb[returnOp.getUnrollDim()] += peelSize;
      shapeOp.updateShape(lb, shapeOp.getUB());
      return success();
    }
    return numOfOperands == 0 ? failure() : success();
  }

  LogicalResult addPeelIteration(stencil::ApplyOp applyOp,
                                 stencil::ReturnOp returnOp, int64_t peelSize,
                                 PatternRewriter &rewriter) const {
    // Get the unroll factor and dimension
    auto unrollFac = returnOp.getUnrollFac();
    auto unrollDim = returnOp.getUnrollDim();

    // Compute the domain size
    auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
    auto domainSize = shapeOp.getUB()[unrollDim] - shapeOp.getLB()[unrollDim];

    // Introduce peel iterations
    if (domainSize <= unrollFac) {
      return makePeelIteration(applyOp, peelSize, rewriter);
    } else {
      // Clone a peel and a body operation
      auto leftOp = cast<stencil::ApplyOp>(rewriter.clone(*applyOp));
      auto rightOp = cast<stencil::ApplyOp>(rewriter.clone(*applyOp));

      // Adapt the shape of the two apply ops
      auto lb = shapeOp.getLB();
      auto ub = shapeOp.getUB();
      int64_t split = peelSize < 0 ? lb[unrollDim] + peelSize + unrollFac
                                   : ub[unrollDim] + peelSize - unrollFac;
      lb[unrollDim] = split;
      ub[unrollDim] = split;

      // Introduce a second apply to handle the peel domain
      cast<ShapeOp>(leftOp.getOperation()).updateShape(shapeOp.getLB(), ub);
      cast<ShapeOp>(rightOp.getOperation()).updateShape(lb, shapeOp.getUB());

      // Remove stores that exceed the domain
      auto peelOp = peelSize < 0 ? leftOp : rightOp;
      makePeelIteration(peelOp, domainSize % unrollFac, rewriter);

      // Introduce a stencil combine to replace the apply operation
      auto combineOp = rewriter.create<stencil::CombineOp>(
          applyOp.getLoc(), applyOp.getResultTypes(), unrollDim, split,
          leftOp.getResults(), rightOp.getResults(), ValueRange(), ValueRange(),
          applyOp.lbAttr(), applyOp.ubAttr());
      rewriter.replaceOp(applyOp, combineOp.getResults());
      return success();
    }
  }

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Get the return operation and the shape of the apply operation
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    auto shapeOp = cast<ShapeOp>(applyOp.getOperation());

    // Compute the domain size
    auto unrollDim = returnOp.getUnrollDim();
    auto unrollFac = returnOp.getUnrollFac();
    if (unrollFac == 1)
      return failure();

    // Get the combine tree root and determine to base offset
    auto rootOp = applyOp.getCombineTreeRootShape();
    auto rootOrigin = rootOp.getLB()[unrollDim];

    // Add left or right peel iterations if the bound is unaligned
    auto leftSize = (shapeOp.getLB()[unrollDim] - rootOrigin) % unrollFac;
    auto rightSize = (shapeOp.getUB()[unrollDim] - rootOrigin) % unrollFac;
    if (leftSize != 0)
      return addPeelIteration(applyOp, returnOp, -leftSize, rewriter);
    if (rightSize != 0)
      return addPeelIteration(applyOp, returnOp, unrollFac - rightSize,
                              rewriter);
    return failure();
  }
};

// Fuse peel loops of neighboring apply operations in unroll direction
struct FuseRewrite : public stencil::CombineOpPattern {
  using CombineOpPattern::CombineOpPattern;

  Operation *getLowerDefiningOp(stencil::CombineOp combineOp) const {
    // Get the lower defining operation
    if (combineOp.lower().empty())
      return nullptr;
    return combineOp.lower().front().getDefiningOp();
  }

  Operation *getUpperDefiningOp(stencil::CombineOp combineOp) const {
    // Get the upper defining operation
    if (combineOp.upper().empty())
      return nullptr;
    return combineOp.upper().front().getDefiningOp();
  }

  // Create an apply operation that fuses the left and right peel iterations
  stencil::ApplyOp fusePeelIterations(stencil::ApplyOp leftOp,
                                      stencil::ApplyOp rightOp,
                                      PatternRewriter &rewriter) const {
    // Compute the operands of the fused apply op
    // (run canonicalization after the pass to cleanup arguments)
    SmallVector<Value, 10> newOperands = leftOp.getOperands();
    newOperands.append(rightOp.getOperands().begin(),
                       rightOp.getOperands().end());

    // Get return operations
    auto leftReturnOp =
        cast<stencil::ReturnOp>(leftOp.getBody()->getTerminator());
    auto rightReturnOp =
        cast<stencil::ReturnOp>(rightOp.getBody()->getTerminator());
    assert(leftReturnOp.unroll() == rightReturnOp.unroll() &&
           "expected unroll of the left and right apply to match");

    // Create a new operation that has the size of
    auto newOp = rewriter.create<stencil::ApplyOp>(
        rewriter.getFusedLoc({leftOp.getLoc(), rightOp.getLoc()}),
        leftOp.getResultTypes(), newOperands, leftOp.lb(), rightOp.ub());
    rewriter.mergeBlocks(
        leftOp.getBody(), newOp.getBody(),
        newOp.getBody()->getArguments().take_front(leftOp.getNumOperands()));
    rewriter.mergeBlocks(
        rightOp.getBody(), newOp.getBody(),
        newOp.getBody()->getArguments().take_back(rightOp.getNumOperands()));

    // Compute the split between left and right op operands
    auto unrollDim = leftReturnOp.getUnrollDim();
    auto unrollFac = leftReturnOp.getUnrollFac();
    unsigned split = cast<ShapeOp>(leftOp.getOperation()).getUB()[unrollDim] -
                     cast<ShapeOp>(leftOp.getOperation()).getLB()[unrollDim];

    // Update the operands of the second return operation
    SmallVector<Value, 10> newReturnOperands = rightReturnOp.getOperands();
    for (auto en : llvm::enumerate(leftReturnOp.getOperands())) {
      if (en.index() % unrollFac < split)
        newReturnOperands[en.index()] = en.value();
    }
    rewriter.updateRootInPlace(rightReturnOp, [&]() {
      rightReturnOp->setOperands(newReturnOperands);
    });
    rewriter.eraseOp(leftReturnOp);

    // Update the shape of the apply operation
    cast<ShapeOp>(newOp.getOperation())
        .updateShape(cast<ShapeOp>(leftOp.getOperation()).getLB(),
                     cast<ShapeOp>(rightOp.getOperation()).getUB());
    return newOp;
  }

  // Introduce a peel loop if the shape is not a multiple of the unroll factor
  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Store the left and right leaf combines
    stencil::CombineOp leftCombineOp = combineOp;
    stencil::CombineOp rightCombineOp = combineOp;

    // Search the lower and upper defining ops and exit if none exists
    Operation *currLeftCombineOp = getLowerDefiningOp(combineOp);
    Operation *currRightCombineOp = getUpperDefiningOp(combineOp);
    if (!currLeftCombineOp || !currRightCombineOp)
      return failure();

    // Walk up the combine tree
    while (auto combineOp =
               dyn_cast_or_null<stencil::CombineOp>(currLeftCombineOp)) {
      leftCombineOp = combineOp;
      currLeftCombineOp = getUpperDefiningOp(combineOp);
    }
    while (auto combineOp =
               dyn_cast_or_null<stencil::CombineOp>(currRightCombineOp)) {
      rightCombineOp = combineOp;
      currRightCombineOp = getLowerDefiningOp(combineOp);
    }

    // Get the left and right apply operations
    auto leftOp = dyn_cast_or_null<stencil::ApplyOp>(currLeftCombineOp);
    auto rightOp = dyn_cast_or_null<stencil::ApplyOp>(currRightCombineOp);
    if (!leftOp || !rightOp)
      return failure();

    // Check if the shapes overlap
    auto returnOp = cast<stencil::ReturnOp>(leftOp.getBody()->getTerminator());
    if (returnOp.getUnrollFac() == 1)
      return failure();
    if (cast<ShapeOp>(leftOp.getOperation()).getLB()[returnOp.getUnrollDim()] !=
        cast<ShapeOp>(rightOp.getOperation()).getLB()[returnOp.getUnrollDim()])
      return failure();

    // Merge the two apply operations in case they overlap
    auto newOp = fusePeelIterations(leftOp, rightOp, rewriter);

    // Disconnect the left and right apply operations from the combine tree
    SmallVector<Value, 10> leftOperands;
    SmallVector<Value, 10> rightOperands;
    if (leftCombineOp != combineOp) {
      rewriter.replaceOp(leftCombineOp, leftCombineOp.lower());
      leftOperands = combineOp.lower();
    }
    if (rightCombineOp != combineOp) {
      rewriter.replaceOp(rightCombineOp, rightCombineOp.upper());
      rightOperands = combineOp.upper();
    }

    // Replace the combine op by the results computed by the fused apply
    SmallVector<Value, 10> newResults = newOp.getResults();
    auto currShape = cast<ShapeOp>(newOp.getOperation());
    auto fullShape = cast<ShapeOp>(combineOp.getOperation());
    auto unrollDim = returnOp.getUnrollDim();
    if (!leftOperands.empty()) {
      // Introduce a combine ob to connect to the left combine subtree
      auto newCombineOp = rewriter.create<stencil::CombineOp>(
          combineOp.getLoc(), combineOp.getResultTypes(), unrollDim,
          currShape.getLB()[unrollDim], leftOperands, newResults, ValueRange(),
          ValueRange(), combineOp.lbAttr(), combineOp.ubAttr());
      newResults = newCombineOp.getResults();

      // Update the shape of the newly introduced combine
      auto lb = fullShape.getLB();
      auto ub = fullShape.getUB();
      ub[unrollDim] = currShape.getUB()[unrollDim];
      currShape = cast<ShapeOp>(newCombineOp.getOperation());
      currShape.updateShape(lb, ub);
    }
    if (!rightOperands.empty()) {
      // Introduce a combine ob to connect to the right combine subtree
      auto newCombineOp = rewriter.create<stencil::CombineOp>(
          combineOp.getLoc(), combineOp.getResultTypes(), unrollDim,
          currShape.getUB()[unrollDim], newResults, rightOperands, ValueRange(),
          ValueRange(), combineOp.lbAttr(), combineOp.ubAttr());
      newResults = newCombineOp.getResults();
    }
    rewriter.replaceOp(combineOp, newResults);
    rewriter.eraseOp(leftOp);
    rewriter.eraseOp(rightOp);
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

  // Check the combine to ifelse preparations have been run
  auto result = funcOp.walk([&](stencil::CombineOp combineOp) {
    if (!combineOp.lowerext().empty() || !combineOp.upperext().empty()) {
      combineOp.emitOpError("expected no lower or upper extra operands");
      return WalkResult::interrupt();
    }
    // Check the producer in the operand range are unique
    auto haveUniqueProducer = [](OperandRange operands) {
      Operation *lastOp = nullptr;
      for (auto operand : operands) {
        auto definingOp = operand.getDefiningOp();
        if (lastOp && definingOp && lastOp != definingOp)
          return false;
        lastOp = definingOp ? definingOp : lastOp;
      }
      return true;
    };
    if (!(haveUniqueProducer(combineOp.lower()) &&
          haveUniqueProducer(combineOp.upper()))) {
      combineOp.emitOpError(
          "expected unique lower and upper producer operations");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();

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
  patterns.insert<PeelRewrite, FuseRewrite>(&getContext());
  applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createPeelOddIterationsPass() {
  return std::make_unique<PeelOddIterationsPass>();
}
