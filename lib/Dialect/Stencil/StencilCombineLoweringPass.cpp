#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>

using namespace mlir;
using namespace stencil;

namespace {

// Pattern replacing a stencil.combine by a single stencil.apply and if/else
struct IfElseRewrite : public OpRewritePattern<stencil::CombineOp> {
  IfElseRewrite(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<stencil::CombineOp>(context, benefit) {}

  LogicalResult lowerStencilCombine(stencil::ApplyOp lowerOp,
                                    stencil::ApplyOp upperOp,
                                    stencil::CombineOp combineOp,
                                    PatternRewriter &rewriter) const {
    auto loc = combineOp.getLoc();
    auto shapeOp = cast<stencil::ShapeOp>(combineOp.getOperation());

    // Compute the operands of the fused apply op
    // (run canonicalization after the pass to cleanup arguments)
    SmallVector<Value, 10> newOperands = lowerOp.getOperands();
    newOperands.insert(newOperands.end(), upperOp.getOperands().begin(),
                       upperOp.getOperands().end());

    // Create a new apply op that updates the lower and upper domains
    // (rerun shape inference after the pass to avoid bound computations)
    auto newOp = rewriter.create<stencil::ApplyOp>(
        loc, newOperands, shapeOp.getLB(), shapeOp.getUB(),
        combineOp.getResultTypes());
    rewriter.setInsertionPointToStart(newOp.getBody());

    // Introduce the branch condition
    SmallVector<int64_t, 3> offset(kIndexSize, 0);
    auto indexOp =
        rewriter.create<stencil::IndexOp>(loc, combineOp.dim(), offset);
    auto constOp = rewriter.create<ConstantOp>(
        loc, rewriter.getIndexAttr(combineOp.index()));
    auto cmpOp =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, indexOp, constOp);

    // Get the return operations and check to unroll factors match
    auto lowerReturnOp =
        cast<stencil::ReturnOp>(lowerOp.getBody()->getTerminator());
    auto upperReturnOp =
        cast<stencil::ReturnOp>(upperOp.getBody()->getTerminator());
    // Check both apply operations have the same unroll configuration if any
    if (lowerReturnOp.getUnrollFactor() != upperReturnOp.getUnrollFactor() ||
        lowerReturnOp.getUnrollDimension() !=
            upperReturnOp.getUnrollDimension()) {
      combineOp.emitWarning("expected matching unroll configurations");
      return failure();
    }

    assert(lowerReturnOp.getOperandTypes() == upperReturnOp.getOperandTypes() &&
           "expected both apply ops to return the same types");
    assert(!lowerReturnOp.getOperandTypes().empty() &&
           "expected apply ops to return at least one value");

    // Introduce the if else op and return the results
    auto ifOp = rewriter.create<scf::IfOp>(loc, lowerReturnOp.getOperandTypes(),
                                           cmpOp, true);
    rewriter.create<stencil::ReturnOp>(loc, ifOp.getResults(),
                                       lowerReturnOp.unroll());

    // Replace the return ops by yield ops
    rewriter.setInsertionPoint(lowerReturnOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(lowerReturnOp,
                                              lowerReturnOp.getOperands());
    rewriter.setInsertionPoint(upperReturnOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(upperReturnOp,
                                              upperReturnOp.getOperands());

    // Move the computation to the new apply operation
    rewriter.mergeBlocks(
        lowerOp.getBody(), ifOp.getBody(0),
        newOp.getBody()->getArguments().take_front(lowerOp.getNumOperands()));
    rewriter.mergeBlocks(
        upperOp.getBody(), ifOp.getBody(1),
        newOp.getBody()->getArguments().take_front(upperOp.getNumOperands()));

    // Remove the combine op and the attached apply ops
    // (assuming the apply ops have not other uses than the combine)
    rewriter.replaceOp(combineOp, newOp.getResults());
    rewriter.eraseOp(upperOp);
    rewriter.eraseOp(lowerOp);
    return success();
  }

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Get the lower and the upper apply up
    auto lowerOp = dyn_cast<stencil::ApplyOp>(combineOp.getLowerOp());
    auto upperOp = dyn_cast<stencil::ApplyOp>(combineOp.getUpperOp());
    if (lowerOp && upperOp) {
      return lowerStencilCombine(lowerOp, upperOp, combineOp, rewriter);
    }
    return failure();
  }
};

struct StencilCombineLoweringPass
    : public StencilCombineLoweringPassBase<StencilCombineLoweringPass> {

  void runOnFunction() override;
};

void StencilCombineLoweringPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!StencilDialect::isStencilProgram(funcOp))
    return;

  // Check shape inference has been executed
  bool hasShapeOpWithoutShape = false;
  funcOp.walk([&](stencil::ShapeOp shapeOp) {
    if (!shapeOp.hasShape())
      hasShapeOpWithoutShape = true;
  });
  if (hasShapeOpWithoutShape) {
    funcOp.emitOpError("execute shape inference before combine lowering");
    signalPassFailure();
    return;
  }

  OwningRewritePatternList patterns;
  patterns.insert<IfElseRewrite>(&getContext());
  applyPatternsAndFoldGreedily(funcOp, patterns);
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createStencilCombineLoweringPass() {
  return std::make_unique<StencilCombineLoweringPass>();
}
