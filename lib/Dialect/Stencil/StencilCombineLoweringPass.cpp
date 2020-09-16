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

  OwningRewritePatternList patterns;
  patterns.insert<IfElseRewrite>(&getContext());
  applyPatternsAndFoldGreedily(funcOp, patterns);
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createStencilCombineLoweringPass() {
  return std::make_unique<StencilCombineLoweringPass>();
}
