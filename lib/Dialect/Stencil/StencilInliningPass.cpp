#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>

using namespace mlir;

namespace {

// Pattern rerouting output edge via consumer
struct RerouteRewrite : public OpRewritePattern<stencil::ApplyOp> {
  RerouteRewrite(MLIRContext *context)
      : OpRewritePattern<stencil::ApplyOp>(context, /*benefit=*/1) {}

  // Helper method inlining the consumer in the producer
  PatternMatchResult redirectStore(stencil::ApplyOp producerOp,
                                   stencil::ApplyOp consumerOp,
                                   PatternRewriter &rewriter) const {
    // Compute operand and result lists
    SmallVector<Value, 10> newOperands = consumerOp.getOperands();
    SmallVector<Value, 10> newResults = consumerOp.getResults();
    for (auto result : producerOp.getResults()) {
      // Count the result uses
      auto uses = result.getUses();
      size_t count = std::distance(uses.begin(), uses.end());
      // Add the result if multiple uses
      if (llvm::is_contained(newOperands, result) && count > 1) {
        newResults.push_back(result);
      }
      // Add parameter and result if not consumed but has uses
      if (!llvm::is_contained(newOperands, result) && count > 0) {
        newOperands.push_back(result);
        consumerOp.getBody()->addArgument(result.getType());
        newResults.push_back(result);
      }
    }

    // Clone the consumer op right after the producer op
    rewriter.setInsertionPointAfter(producerOp);
    auto loc = consumerOp.getLoc();
    auto newOp =
        rewriter.create<stencil::ApplyOp>(loc, newOperands, newResults);
    rewriter.cloneRegionBefore(consumerOp.region(), newOp.region(),
                               newOp.region().begin());

    // Get the terminator of the cloned consumer op
    auto returnOp =
        dyn_cast<stencil::ReturnOp>(newOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(returnOp);

    // Add access to load the rerouted parameters
    SmallVector<Value, 10> returnOperands = returnOp.getOperands();
    for (auto result : producerOp.getResults()) {
      if (llvm::is_contained(newResults, result)) {
        // Compute the argument index and add an access op
        auto it = llvm::find(newOperands, result);
        size_t index =
            std::distance(newOperands.begin(), llvm::find(newOperands, result));
        SmallVector<int64_t, 3> zeroOffset = {0, 0, 0};
        auto accessOp = rewriter.create<stencil::AccessOp>(
            loc, newOp.getBody()->getArgument(index), zeroOffset);
        returnOperands.push_back(accessOp.getResult());
      }
    }

    // Replace the return op
    rewriter.create<stencil::ReturnOp>(loc, returnOperands);
    rewriter.eraseOp(returnOp);

    // Replace all producer and consumer results
    for (size_t i = 0, e = consumerOp.getResults().size(); i != e; ++i) {
      consumerOp.getResult(i).replaceAllUsesWith(newOp.getResult(i));
    }
    for (auto result : producerOp.getResults()) {
      auto it = llvm::find(newResults, result);
      if (it != newResults.end()) {
        // Store the operand index
        size_t index = std::distance(newOp.operands().begin(),
                                     llvm::find(newOp.operands(), result));
        result.replaceAllUsesWith(
            newOp.getResult(std::distance(newResults.begin(), it)));
        // Restore the operand after replacing all uses
        newOp.setOperand(index, result);
      }
    }

    // Remove the consumer op
    rewriter.eraseOp(consumerOp);
    return matchFailure();
  }

  PatternMatchResult matchAndRewrite(stencil::ApplyOp applyOp,
                                     PatternRewriter &rewriter) const override {
    // Search consumer connected to a single producer
    SmallVector<Operation *, 10> producerOps;
    for (auto operand : applyOp.operands()) {
      if (isa_and_nonnull<stencil::ApplyOp>(operand.getDefiningOp())) {
        if (!llvm::is_contained(producerOps, operand.getDefiningOp()))
          producerOps.push_back(operand.getDefiningOp());
      }
    }

    // Redirect outputs of the producer
    if (producerOps.size() == 1) {
      // TODO we may want to ensure that producer has multiple consumers
      // (however as long as the inlining pattern has a higher benefit this is
      // not needed)
      return redirectStore(cast<stencil::ApplyOp>(producerOps.front()), applyOp,
                           rewriter);
    }
    return matchFailure();
  }
};

// Pattern inlining producer into consumer
// (assuming the producer has only a single consumer)
struct InliningRewrite : public OpRewritePattern<stencil::ApplyOp> {
  InliningRewrite(MLIRContext *context)
      : OpRewritePattern<stencil::ApplyOp>(context, /*benefit=*/2) {}

  // Helper method replacing all uses of temporary by inline computation
  void replaceAccess(stencil::ApplyOp consumerOp, stencil::AccessOp accessOp,
                     ValueRange producerResults,
                     ValueRange computedResults) const {
    for (unsigned i = 0, e = consumerOp.getNumOperands(); i != e; ++i) {
      if (consumerOp.getBody()->getArgument(i) == accessOp.view()) {
        size_t index = std::distance(
            producerResults.begin(),
            llvm::find(producerResults, consumerOp.getOperand(i)));
        assert(index < computedResults.size() &&
               "failed to find inlined computation");
        accessOp.getResult().replaceAllUsesWith(computedResults[index]);
        break;
      }
    }
  }

  // Helper method inlining the producer computation
  PatternMatchResult inlineProducer(stencil::ApplyOp producerOp,
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
    auto newOp = rewriter.create<stencil::ApplyOp>(loc, newOperands,
                                                   consumerOp.getResults());
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
      if (llvm::count(newOp.getBody()->getArguments(), accessOp.view()) == 0) {
        SmallVector<int64_t, 3> offset = accessOp.getOffset();
        // Clone the producer and shift the offsets
        auto clonedOp = producerOp.getOperation()->clone(mapper);
        clonedOp->walk([&](stencil::AccessOp accessOp) {
          SmallVector<int64_t, 3> sum(offset.size());
          llvm::transform(llvm::zip(offset, accessOp.getOffset()), sum.begin(),
                          [](std::tuple<int64_t, int64_t> x) {
                            return std::get<0>(x) + std::get<1>(x);
                          });
          accessOp.setOffset(sum);
        });

        // Copy the operations in after the access op
        newOp.getBody()->getOperations().splice(
            Block::iterator{accessOp},
            clonedOp->getRegion(0).front().getOperations());
        clonedOp->erase();

        // Replace all uses of the accesOp
        stencil::ReturnOp returnOp =
            cast<stencil::ReturnOp>(*std::prev(Block::iterator(accessOp)));
        replaceAccess(consumerOp, accessOp, producerResults,
                      returnOp.getOperands());
        rewriter.eraseOp(returnOp);
        rewriter.eraseOp(accessOp);
      }
    });

    // Update the all uses and copy the loop bounds
    for (size_t i = 0, e = consumerOp.getResults().size(); i != e; ++i) {
      consumerOp.getResult(i).replaceAllUsesWith(newOp.getResult(i));
    }

    // Erase the producer and consumer ops
    rewriter.eraseOp(consumerOp);
    rewriter.eraseOp(producerOp);
    return matchSuccess();
  }

  PatternMatchResult matchAndRewrite(stencil::ApplyOp applyOp,
                                     PatternRewriter &rewriter) const override {
    // Search producer apply op
    for (auto operand : applyOp.operands()) {
      if (isa_and_nonnull<stencil::ApplyOp>(operand.getDefiningOp())) {
        // Check if multiple consumers
        auto producerResults = operand.getDefiningOp()->getResults();
        for (auto result : producerResults) {
          if (llvm::any_of(result.getUsers(), [&](Operation *op) {
                return op != applyOp.getOperation();
              }))
            return matchFailure();
        }

        // If there is only a single consumer perform the inlining
        return inlineProducer(cast<stencil::ApplyOp>(operand.getDefiningOp()),
                              applyOp, producerResults, rewriter);
      }
    }
    return matchFailure();
  }
};

struct StencilInliningPass : public FunctionPass<StencilInliningPass> {
  void runOnFunction() override;
};

void StencilInliningPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  OwningRewritePatternList patterns;
  patterns.insert<InliningRewrite, RerouteRewrite>(&getContext());
  applyPatternsGreedily(funcOp, patterns);
}

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::stencil::createStencilInliningPass() {
  return std::make_unique<StencilInliningPass>();
}

static PassRegistration<StencilInliningPass> pass("stencil-inlining",
                                                  "Inline stencil apply ops");
