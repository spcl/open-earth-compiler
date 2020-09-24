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
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;
using namespace stencil;

namespace {

// // Base class of all storage materialization patterns
// struct StorageMaterializationPattern
//     : public OpRewritePattern<stencil::ApplyOp> {
//   StorageMaterializationPattern(MLIRContext *context,
//                                 PatternBenefit benefit = 1)
//       : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}
// };

// Pattern introducing buffers between consecutive apply ops
struct ApplyOpRewrite : public OpRewritePattern<stencil::ApplyOp> {
  ApplyOpRewrite(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}

  // Introduce buffer on the output edge connected to another apply
  LogicalResult introduceOutputBuffers(stencil::ApplyOp applyOp,
                                       PatternRewriter &rewriter) const {
    auto loc = applyOp.getLoc();

    // Create another apply operation and move the body
    auto newOp = rewriter.create<stencil::ApplyOp>(
        loc, applyOp.getResultTypes(), applyOp.getOperands(), applyOp.lb(),
        applyOp.ub());
    rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(),
                         newOp.getBody()->getArguments());

    // Introduce a buffer on every result connected to another result
    SmallVector<Value, 10> repResults = newOp.getResults();
    for (auto result : applyOp.getResults()) {
      if (llvm::any_of(result.getUsers(), [](Operation *op) {
            return isa<stencil::ApplyOp>(op);
          })) {
        auto bufferOp = rewriter.create<stencil::BufferOp>(
            loc, result.getType(), newOp.getResult(result.getResultNumber()),
            nullptr, nullptr);
        repResults[result.getResultNumber()] = bufferOp;
      }
    }
    rewriter.replaceOp(applyOp, repResults);
    return success();
  }

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(applyOp.getOperation()->getUsers(),
                     [](Operation *op) { return isa<stencil::ApplyOp>(op); }))
      return introduceOutputBuffers(applyOp, rewriter);
    return failure();
  }
};

struct StorageMaterializationPass
    : public StorageMaterializationPassBase<StorageMaterializationPass> {

  void runOnFunction() override;
};

void StorageMaterializationPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!StencilDialect::isStencilProgram(funcOp))
    return;

  // Poppulate the pattern list depending on the config
  OwningRewritePatternList patterns;
  patterns.insert<ApplyOpRewrite>(&getContext());
  applyPatternsAndFoldGreedily(funcOp, patterns);
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createStorageMaterializationPass() {
  return std::make_unique<StorageMaterializationPass>();
}
