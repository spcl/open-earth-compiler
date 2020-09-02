#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <bits/stdint-intn.h>
#include <cstddef>
#include <cstdint>
#include <iterator>

using namespace mlir;
using namespace stencil;

namespace {

// Base class for the stencil inlining patterns
struct StencilInliningPattern : public OpRewritePattern<stencil::ApplyOp> {
  StencilInliningPattern(MLIRContext *context)
      : OpRewritePattern<stencil::ApplyOp>(context, /*benefit=*/2) {}

  // Check if the the apply operation is the only consumer
  bool hasSingleConsumer(stencil::ApplyOp producerOp,
                         stencil::ApplyOp applyOp) const {
    return llvm::all_of(producerOp.getOperation()->getUsers(),
                        [&](Operation *op) { return op == applyOp; });
  }

  // Check if inlining is imposible at all
  bool isStencilInliningImpossible(stencil::ApplyOp producerOp,
                                   stencil::ApplyOp consumerOp) const {
    // Do not inline sequential producers or consumers currently
    if (producerOp.seq().hasValue() || consumerOp.seq().hasValue())
      return true;

    // Do not inline producers accessed at dynamic offsets
    for (auto operand : llvm::enumerate(consumerOp.operands())) {
      if (operand.value().getDefiningOp() == producerOp &&
          llvm::any_of(
              consumerOp.getBody()->getArgument(operand.index()).getUsers(),
              [](Operation *op) { return isa<stencil::DynAccessOp>(op); }))
        return true;
    }

    return false;
  }

  // Check if rerouting is impossible
  bool isStencilReroutingImpossible(stencil::ApplyOp producerOp,
                                    stencil::ApplyOp consumerOp) const {
    // Ensure the producer has not a single use otherwise inlining applies
    if (hasSingleConsumer(producerOp, consumerOp))
      return true;

    // Ensure the consumer dependencies are computed before the producer
    for (auto operand : consumerOp.getOperands()) {
      if (operand.getDefiningOp()) {
        if (producerOp.getOperation()->isBeforeInBlock(operand.getDefiningOp()))
          return true;
      }
    }

    return false;
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
    SmallVector<Value, 10> newResults = consumerOp.getResults();
    unsigned numOfReroutes = 0;
    for (auto results :
         llvm::zip(producerOp.getResults(), clonedOp.getResults())) {
      Value original, cloned;
      std::tie(original, cloned) = results;
      // Add the results that have uses to the consumer results
      if (llvm::any_of(original.getUsers(),
                       [&](Operation *op) { return op != consumerOp; })) {
        newResults.push_back(cloned);
        newOperands.push_back(cloned);
        numOfReroutes++;
      }
      // Replace the producer of result in the operands
      llvm::transform(newOperands, newOperands.begin(), [&](Value value) {
        return value == original ? cloned : value;
      });
    }

    // Create new consumer op right after the producer op
    auto newOp = rewriter.create<stencil::ApplyOp>(
        consumerOp.getLoc(), newOperands, newResults, None);
    rewriter.mergeBlocks(consumerOp.getBody(), newOp.getBody(),
                         newOp.getBody()->getArguments().take_front(
                          consumerOp.getNumOperands()));

    // Get the terminator of the cloned consumer op
    auto returnOp =
        dyn_cast<stencil::ReturnOp>(newOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(returnOp);

    // Add accesses to rerouted operands and replace the return op
    SmallVector<Value, 10> retOperands = returnOp.getOperands();
    for (auto arg : newOp.getBody()->getArguments().take_back(numOfReroutes)) {
      Index zeroOffset = {0, 0, 0};
      retOperands.push_back(rewriter.create<stencil::AccessOp>(
          returnOp.getLoc(), arg, zeroOffset));
    }
    rewriter.create<stencil::ReturnOp>(returnOp.getLoc(), retOperands, nullptr);
    rewriter.eraseOp(returnOp);

    // Replace the producer and consumer ops
    rewriter.replaceOp(producerOp, newOp.getResults().take_back(numOfReroutes));
    rewriter.replaceOp(
        consumerOp, newOp.getResults().take_front(consumerOp.getNumResults()));
    return success();
  }

  // Search a matching producer that can be rerouted
  stencil::ApplyOp searchMatchingProducer(stencil::ApplyOp applyOp) const {
    // Search a producer that can be rerouted
    for (auto operand : applyOp.operands()) {
      if (auto producerOp =
              dyn_cast_or_null<stencil::ApplyOp>(operand.getDefiningOp())) {
        if (isStencilReroutingImpossible(producerOp, applyOp) ||
            isStencilInliningImpossible(producerOp, applyOp))
          continue;

        return producerOp;
      }
    }

    return nullptr;
  }

  // Search a matching consumer of the same input
  stencil::ApplyOp searchMatchingConsumer(stencil::ApplyOp applyOp) const {
    // Search a consumer that shares the same input
    for (auto operand : applyOp.operands()) {
      if (operand.getDefiningOp() &&
          !isa<stencil::ApplyOp>(operand.getDefiningOp())) {
        for (auto user : operand.getDefiningOp()->getUsers()) {
          // Only consider other consumers
          if (user == applyOp.getOperation())
            continue;

          // Only consider users before the apply
          if (!user->isBeforeInBlock(applyOp))
            continue;

          // Only consider other apply operations
          if (auto consumerOp = dyn_cast<stencil::ApplyOp>(user)) {
            if (isStencilReroutingImpossible(consumerOp, applyOp) ||
                isStencilInliningImpossible(consumerOp, applyOp))
              continue;

            return consumerOp;
          }
        }
      }
    }

    return nullptr;
  }

  // Find a match and reroute the outputs of the stencil apply
  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {

    // Only one should be needed
    // // Try to reroute the outputs of matching producer
    // if (auto producerOp = searchMatchingProducer(applyOp)) {
    //   return redirectStore(producerOp, applyOp, rewriter);
    // }

    // Try to reroute the outputs of consumer that shares an input
    if (auto producerOp = searchMatchingConsumer(applyOp)) {
      return redirectStore(producerOp, applyOp, rewriter);
    }

    return failure();
  }
};

// Pattern inlining producer into consumer
// (assuming the producer has only a single consumer)
struct InliningRewrite : public StencilInliningPattern {
  using StencilInliningPattern::StencilInliningPattern;

  // Helper method replacing all uses of temporary by inline computation
  void replaceAccess(stencil::ApplyOp consumerOp, stencil::AccessOp accessOp,
                     ValueRange producerResults, stencil::ReturnOp returnOp,
                     PatternRewriter &rewriter) const {
    for (unsigned i = 0, e = consumerOp.getNumOperands(); i != e; ++i) {
      if (consumerOp.getBody()->getArgument(i) == accessOp.temp()) {
        size_t index = std::distance(
            producerResults.begin(),
            llvm::find(producerResults, consumerOp.getOperand(i)));
        assert(index < returnOp.getNumOperands() &&
               "failed to find inlined computation");
        rewriter.replaceOp(accessOp, returnOp.getOperand(index));
        break;
      }
    }
  }

  // Helper method inlining the producer computation
  LogicalResult inlineProducer(stencil::ApplyOp producerOp,
                               stencil::ApplyOp consumerOp,
                               ValueRange producerResults,
                               PatternRewriter &rewriter) const {
    // Compute the operand list and an argument mapper befor cloning
    BlockAndValueMapping mapper;
    SmallVector<Value, 10> newOperands;
    for (unsigned i = 0, e = consumerOp.getNumOperands(); i != e; ++i) {
      if (llvm::is_contained(producerResults, consumerOp.getOperand(i)))
        mapper.map(consumerOp.getBody()->getArgument(i),
                   consumerOp.getBody()->getArgument(i));
      else
        newOperands.push_back(consumerOp.getOperand(i));
    }
    for (auto operand : producerOp.getOperands()) {
      if (!llvm::is_contained(newOperands, operand)) {
        newOperands.push_back(operand);
        consumerOp.getBody()->addArgument(operand.getType());
      }
    }

    // Clone the consumer op
    auto loc = consumerOp.getLoc();
    auto newOp = rewriter.create<stencil::ApplyOp>(
        loc, newOperands, consumerOp.getResults(), None);
    rewriter.cloneRegionBefore(consumerOp.region(), newOp.region(),
                               newOp.region().begin(), mapper);

    // Add mappings for the producer operands
    for (unsigned i = 0, e = producerOp.getNumOperands(); i != e; ++i) {
      auto it = llvm::find(newOperands, producerOp.getOperand(i));
      assert(it != newOperands.end() && "expected to find producer operand");
      mapper.map(
          producerOp.getBody()->getArgument(i),
          newOp.getBody()->getArgument(std::distance(newOperands.begin(), it)));
    }

    // Walk accesses of producer results and replace them by computation
    newOp.walk([&](stencil::AccessOp accessOp) {
      if (llvm::count(newOp.getBody()->getArguments(), accessOp.temp()) == 0) {
        auto offsetOp = cast<OffsetOp>(accessOp.getOperation());
        Index offset = offsetOp.getOffset();
        // Copy the operations in after the access op
        rewriter.setInsertionPoint(accessOp);
        for (auto &op : producerOp.getBody()->getOperations()) {
          auto clonedOp = rewriter.clone(op, mapper);
          clonedOp->walk([&](stencil::OffsetOp offsetOp) {
            offsetOp.setOffset(applyFunElementWise(offsetOp.getOffset(), offset,
                                                   std::plus<int64_t>()));
          });
        }

        // Replace all uses of the accesOp
        stencil::ReturnOp returnOp =
            cast<stencil::ReturnOp>(*std::prev(Block::iterator(accessOp)));
        replaceAccess(consumerOp, accessOp, producerResults, returnOp,
                      rewriter);
        rewriter.eraseOp(returnOp);
      }
    });

    // Update the all uses and copy the loop bounds
    rewriter.replaceOp(consumerOp, newOp.getResults());
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
        if (isStencilInliningImpossible(producerOp, applyOp))
          continue;

        // Inline if the producer has a single consumer
        if (hasSingleConsumer(producerOp, applyOp)) {
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
      returnOp.emitOpError("execute stencil inlining after stencil unrolling");
      hasUnrolledStencils = true;
    }
  });
  if (hasUnrolledStencils) {
    signalPassFailure();
    return;
  }

  OwningRewritePatternList patterns;
  patterns.insert<InliningRewrite, RerouteRewrite>(&getContext());
  applyPatternsAndFoldGreedily(funcOp, patterns);
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createStencilInliningPass() {
  return std::make_unique<StencilInliningPass>();
}
