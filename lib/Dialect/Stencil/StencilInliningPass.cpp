#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace stencil;

namespace {

// Base class for the stencil inlining patterns
struct StencilInliningPattern : public ApplyOpPattern {
  StencilInliningPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : ApplyOpPattern(context, benefit){};

  // Check if the the apply operation is the only consumer
  bool hasSingleConsumer(stencil::ApplyOp producerOp,
                         stencil::ApplyOp applyOp) const {
    return llvm::all_of(producerOp.getOperation()->getUsers(),
                        [&](Operation *op) { return op == applyOp; });
  }

  // Check if inlining is possible
  bool isStencilInliningPossible(stencil::ApplyOp producerOp,
                                 stencil::ApplyOp consumerOp) const {
    // Do not inline producer ops that return void values
    bool containsEmptyStores = false;
    producerOp.walk([&](stencil::StoreResultOp resultOp) {
      if (resultOp.operands().size() == 0)
        containsEmptyStores = true;
    });
    if (containsEmptyStores)
      return false;

    // Do not inline producers accessed at dynamic offsets
    for (auto operand : llvm::enumerate(consumerOp.operands())) {
      if (operand.value().getDefiningOp() == producerOp &&
          llvm::any_of(
              consumerOp.getBody()->getArgument(operand.index()).getUsers(),
              [](Operation *op) { return isa<stencil::DynAccessOp>(op); }))
        return false;
    }
    return true;
  }

  // Check if rerouting is possible
  bool isStencilReroutingPossible(stencil::ApplyOp producerOp,
                                  stencil::ApplyOp consumerOp) const {
    // Perform producer consumer inlining instead
    if (hasSingleConsumer(producerOp, consumerOp))
      return false;

    // Ensure the consumer dependencies are computed before the producer
    for (auto operand : consumerOp.getOperands()) {
      if (operand.getDefiningOp()) {
        if (producerOp.getOperation()->isBeforeInBlock(operand.getDefiningOp()))
          return false;
      }
    }
    return true;
  }
};

// Pattern rerouting output edge via consumer
struct RerouteRewrite : public StencilInliningPattern {
  using StencilInliningPattern::StencilInliningPattern;

  // Helper method inlining the consumer in the producer
  LogicalResult redirectStore(stencil::ApplyOp producerOp,
                              stencil::ApplyOp consumerOp,
                              PatternRewriter &rewriter) const {
    // Clone the producer op
    rewriter.setInsertionPointAfter(producerOp);
    auto clonedOp = rewriter.cloneWithoutRegions(producerOp);
    rewriter.inlineRegionBefore(producerOp.region(), clonedOp.region(),
                                clonedOp.region().begin());

    // Compute operand and result lists for the new consumer
    SmallVector<Value, 10> newOperands = consumerOp.getOperands();
    SmallVector<Type, 10> newResultTypes(consumerOp.getResultTypes().begin(),
                                         consumerOp.getResultTypes().end());
    unsigned rerouteCount = 0;
    for (auto results :
         llvm::zip(producerOp.getResults(), clonedOp.getResults())) {
      Value original, cloned;
      std::tie(original, cloned) = results;
      // Add the results that have uses to the consumer results
      if (llvm::any_of(original.getUsers(),
                       [&](Operation *op) { return op != consumerOp; })) {
        newResultTypes.push_back(cloned.getType());
        newOperands.push_back(cloned);
        rerouteCount++;
      }
      // Replace the producer of result in the operands
      llvm::transform(newOperands, newOperands.begin(), [&](Value value) {
        return value == original ? cloned : value;
      });
    }

    // Create new consumer op right after the producer op
    auto newOp = rewriter.create<stencil::ApplyOp>(
        consumerOp.getLoc(), newResultTypes, newOperands, consumerOp.lb(),
        consumerOp.ub());
    rewriter.mergeBlocks(consumerOp.getBody(), newOp.getBody(),
                         newOp.getBody()->getArguments().take_front(
                             consumerOp.getNumOperands()));

    // Extend the size of the consumer after the rerouting
    auto producerShape = cast<ShapeOp>(producerOp.getOperation());
    auto consumerShape = cast<ShapeOp>(consumerOp.getOperation());
    auto newShape = cast<ShapeOp>(newOp.getOperation());
    if (producerShape.hasShape() && consumerShape.hasShape()) {
      auto lb = applyFunElementWise(producerShape.getLB(),
                                    consumerShape.getLB(), min);
      auto ub = applyFunElementWise(producerShape.getUB(),
                                    consumerShape.getUB(), max);
      newShape.updateShape(lb, ub);
      // Update the region arguments of dependent shape ops
      for (auto user : newShape.getOperation()->getUsers()) {
        if (auto shapeOp = dyn_cast<ShapeOp>(user))
          shapeOp.updateArgumentTypes();
      }
    }

    // Get the terminator of the cloned consumer op
    auto returnOp =
        dyn_cast<stencil::ReturnOp>(newOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(returnOp);

    // Add accesses to rerouted operands and replace the return op
    SmallVector<Value, 10> retOperands = returnOp.getOperands();
    for (auto arg : newOp.getBody()->getArguments().take_back(rerouteCount)) {
      Index zeroOffset = {0, 0, 0};
      auto resultOp = rewriter.create<stencil::StoreResultOp>(
          returnOp.getLoc(), rewriter.create<stencil::AccessOp>(
                                 returnOp.getLoc(), arg, zeroOffset));
      retOperands.push_back(resultOp);
    }
    rewriter.create<stencil::ReturnOp>(returnOp.getLoc(), retOperands, nullptr);
    rewriter.eraseOp(returnOp);

    // Compute the replacement values for the producer results
    SmallVector<Value, 10> repResults =
        newOp.getResults().take_back(rerouteCount);
    // Duplicate the last element until we have enough replacement values
    while (repResults.size() < producerOp.getNumResults())
      repResults.push_back(repResults.back());

    // Replace the producer and consumer ops
    rewriter.replaceOp(producerOp, repResults);
    rewriter.replaceOp(
        consumerOp, newOp.getResults().take_front(consumerOp.getNumResults()));
    return success();
  }

  // Find a match and reroute the outputs of the stencil apply
  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Reroute input dependency
    for (auto operand : applyOp.operands()) {
      if (operand.getDefiningOp()) {
        for (auto user : operand.getDefiningOp()->getUsers()) {
          // Only consider other apply operations
          if (auto producerOp = dyn_cast<stencil::ApplyOp>(user)) {
            // Only consider other consumers before the apply op
            if (user == applyOp.getOperation() ||
                !user->isBeforeInBlock(applyOp))
              continue;

            if (isStencilInliningPossible(producerOp, applyOp) &&
                isStencilReroutingPossible(producerOp, applyOp))
              return redirectStore(producerOp, applyOp, rewriter);
          }
        }
      }
    }
    // Reroute output dependency
    for (auto operand : applyOp.operands()) {
      if (auto producerOp =
              dyn_cast_or_null<stencil::ApplyOp>(operand.getDefiningOp())) {
        if (isStencilInliningPossible(producerOp, applyOp) &&
            isStencilReroutingPossible(producerOp, applyOp))
          return redirectStore(producerOp, applyOp, rewriter);
      }
    }
    return failure();
  }
};

// Pattern inlining producer into consumer
// (assuming the producer has only a single consumer)
struct InliningRewrite : public StencilInliningPattern {
  using StencilInliningPattern::StencilInliningPattern;

  // Helper method inlining the producer computation
  LogicalResult inlineProducer(stencil::ApplyOp producerOp,
                               stencil::ApplyOp consumerOp,
                               ValueRange producerResults,
                               PatternRewriter &rewriter) const {
    // Concatenate the operands of producer and consumer
    SmallVector<Value, 10> buildOperands = producerOp.getOperands();
    buildOperands.append(consumerOp.getOperands().begin(),
                         consumerOp.getOperands().end());

    // Create a build op to assemble the body of the inlined stencil
    auto loc = consumerOp.getLoc();
    auto buildOp = rewriter.create<stencil::ApplyOp>(
        loc, consumerOp.getResultTypes(), buildOperands, consumerOp.lb(),
        consumerOp.ub());
    rewriter.mergeBlocks(consumerOp.getBody(), buildOp.getBody(),
                         buildOp.getBody()->getArguments().take_back(
                             consumerOp.getNumOperands()));

    // Compute the producer result arguments and the replacement offset
    DenseMap<Value, size_t> replacementIndex;
    for (auto en : llvm::enumerate(buildOperands)) {
      auto pos = std::find(producerOp.getResults().begin(),
                           producerOp.getResults().end(), en.value());
      if (pos != producerOp.getResults().end()) {
        replacementIndex[buildOp.getBody()->getArgument(en.index())] =
            std::distance(producerOp.getResults().begin(), pos);
      }
    }

    // Remove the store results ops in the producer to make it inlineable
    producerOp.walk([&](Operation *op) {
      // Remove the store result ops
      if (auto resultOp = dyn_cast<stencil::StoreResultOp>(op)) {
        assert(resultOp.operands().size() == 1 &&
               "expected store result ops to store a value");
        rewriter.replaceOp(resultOp, resultOp.operands());
      }
      // Remove the result types from the signature of the if ops.
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (llvm::any_of(ifOp.getResultTypes(),
                         [](Type type) { return type.isa<ResultType>(); })) {
          SmallVector<Type, 10> newTypes;
          llvm::transform(ifOp.getResultTypes(), std::back_inserter(newTypes),
                          [](Type type) {
                            if (auto resultType = type.dyn_cast<ResultType>())
                              return resultType.getResultType();
                            return type;
                          });
          rewriter.setInsertionPoint(ifOp);
          auto newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newTypes,
                                                  ifOp.condition(), true);
          // All if operations returning a result have both results.
          rewriter.mergeBlocks(ifOp.getBody(0), newOp.getBody(0));
          rewriter.mergeBlocks(ifOp.getBody(1), newOp.getBody(1));
          rewriter.replaceOp(ifOp, newOp.getResults());
        }
      }
    });

    // Walk accesses of producer results and replace them by computation
    DenseMap<Value, SmallVector<std::tuple<Index, Value>, 10>> inliningCache;
    rewriter.setInsertionPoint(buildOp);
    buildOp.walk([&](stencil::AccessOp accessOp) {
      if (replacementIndex.count(accessOp.temp()) != 0) {
        // Get the shift offset
        Index offset = cast<OffsetOp>(accessOp.getOperation()).getOffset();
        // Check if the given producer offset has been inlined before
        if (inliningCache.count(accessOp.temp()) != 0) {
          for (auto it : inliningCache[accessOp.temp()]) {
            if (std::get<0>(it) == offset &&
                std::get<1>(it).getParentRegion()->isAncestor(
                    accessOp.getParentRegion())) {
              rewriter.replaceOp(accessOp, std::get<1>(it));
              return;
            }
          }
        }
        // Otherwise clone the producer in place and shift the offsets
        auto clonedOp = cast<stencil::ApplyOp>(rewriter.clone(*producerOp));
        clonedOp.walk(
            [&](stencil::ShiftOp shiftOp) { shiftOp.shiftByOffset(offset); });
        // Merge into to build op and erase the clone
        rewriter.mergeBlockBefore(clonedOp.getBody(), accessOp,
                                  buildOp.getBody()->getArguments().take_front(
                                      producerOp.getNumOperands()));
        rewriter.eraseOp(clonedOp);
        // Replace the access operation by the result of return operation
        auto returnOp =
            cast<stencil::ReturnOp>(accessOp.getOperation()->getPrevNode());
        auto operand = returnOp.getOperand(replacementIndex[accessOp.temp()]);
        rewriter.replaceOp(accessOp, operand);
        rewriter.eraseOp(returnOp);
        // Cache the result of the inlined producer
        inliningCache[accessOp.temp()].push_back(
            std::make_tuple(offset, operand));
      }
    });

    // Clean unused and duplicate arguments of the build op
    auto newOp = cleanupOpArguments(buildOp, rewriter);
    assert(newOp && "expected op to have unused producer consumer edges");

    // Update the all uses and cleanup temporary ops
    rewriter.replaceOp(consumerOp, newOp.getResults());
    rewriter.eraseOp(buildOp);
    rewriter.eraseOp(producerOp);
    return success();
  }

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Search producer apply op
    for (auto operand : applyOp.operands()) {
      if (auto producerOp =
              dyn_cast_or_null<stencil::ApplyOp>(operand.getDefiningOp())) {
        // Try the next producer if inlining the current one is not possible
        if (isStencilInliningPossible(producerOp, applyOp) &&
            hasSingleConsumer(producerOp, applyOp)) {
          return inlineProducer(producerOp, applyOp, producerOp.getResults(),
                                rewriter);
        }
      }
    }
    return failure();
  }
};

struct StencilInliningPass
    : public StencilInliningPassBase<StencilInliningPass> {
  void runOnFunction() override;
};

void StencilInliningPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  // Only run on functions marked as stencil programs
  if (!StencilDialect::isStencilProgram(funcOp))
    return;

  // Verify unrolling has not been executed
  bool hasUnrolledStencils = false;
  funcOp.walk([&](stencil::ReturnOp returnOp) {
    if (returnOp.unroll().hasValue()) {
      returnOp.emitOpError("execute stencil unrolling after stencil inlining");
      hasUnrolledStencils = true;
    }
  });
  if (hasUnrolledStencils) {
    signalPassFailure();
    return;
  }

  OwningRewritePatternList patterns;
  patterns.insert<InliningRewrite, RerouteRewrite>(&getContext());
  applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createStencilInliningPass() {
  return std::make_unique<StencilInliningPass>();
}
