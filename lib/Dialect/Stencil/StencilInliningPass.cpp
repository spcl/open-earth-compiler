#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
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
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>

using namespace mlir;

namespace {

// Pattern redirecting output edge via consumer
// (assuming the producer has only a single consumer)
struct RedirectRewrite : public OpRewritePattern<stencil::ApplyOp> {
  RedirectRewrite(MLIRContext *context)
      : OpRewritePattern<stencil::ApplyOp>(context, /*benefit=*/1) {}

  // Helper method inlining the producer computation
  PatternMatchResult redirectStore(stencil::ApplyOp producerOp,
                                   stencil::ApplyOp consumerOp,
                                   stencil::StoreOp storeOp,
                                   PatternRewriter &rewriter) const {
    // Add additional identity node that reads all producer and consumer inputs

    // helper to merge operand lists?

    // Add access the additional parameter and return the result

    return matchFailure();
  }

  PatternMatchResult matchAndRewrite(stencil::ApplyOp applyOp,
                                     PatternRewriter &rewriter) const override {
    // Match a producer consumer pair that stores producer outputs
    for (auto operand : applyOp.operands()) {
      if (auto producerOp =
              dyn_cast<stencil::ApplyOp>(operand->getDefiningOp())) {
        // Check one consumer and output
        bool singleConsumer = true;
        Operation *storeOp = nullptr;
        for (auto result : producerOp.getResults()) {
          for (auto user : result.getUsers()) {
            if (isa<stencil::ApplyOp>(user) && user != applyOp.getOperation())
              singleConsumer = false;
            if (isa<stencil::StoreOp>(user))
              storeOp = user;
          }
        }
        // Redirect producer write
        if (storeOp && singleConsumer) {
          return redirectStore(producerOp, applyOp,
                               cast<stencil::StoreOp>(storeOp), rewriter);
        }
      }
    }
    return matchFailure();
  }
};

// Pattern inlining producer into consumer
// (assuming the producer has only a single consumer)
struct ProducerInliningRewrite : public OpRewritePattern<stencil::ApplyOp> {
  ProducerInliningRewrite(MLIRContext *context)
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
    for (unsigned i = 0, e = producerOp.getNumOperands(); i != e; ++i) {
      if (!llvm::is_contained(newOperands, producerOp.getOperand(i))) {
        newOperands.push_back(producerOp.getOperand(i));
        consumerOp.getBody()->addArgument(producerOp.getOperand(i).getType());
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

    // Cache results for previously inlined offsets
    DenseMap<ArrayRef<int64_t>, ValueRange> offsetCache;

    // Walk accesses of producer results and replace them by computation
    newOp.walk([&](stencil::AccessOp accessOp) {
      if (llvm::count(newOp.getBody()->getArguments(), accessOp.view()) == 0) {
        ArrayRef<int64_t> offset = accessOp.getOffset();
        if (offsetCache.count(offset) != 0) {
          // Used cached values if possible
          replaceAccess(consumerOp, accessOp, producerResults,
                        offsetCache[offset]);
        } else {
          // Clone the producer and shift the offsets
          auto clonedOp = producerOp.getOperation()->clone(mapper);
          clonedOp->walk([&](stencil::AccessOp accessOp) {
            SmallVector<int64_t, 3> sum(offset.size());
            llvm::transform(llvm::zip(offset, accessOp.getOffset()),
                            sum.begin(), [](std::tuple<int64_t, int64_t> x) {
                              return std::get<0>(x) + std::get<1>(x);
                            });
            accessOp.setOffset(sum);
          });

          // Copy the operations in after the access op and erase the cloned op
          newOp.getBody()->getOperations().splice(
              Block::iterator{accessOp},
              clonedOp->getRegion(0).front().getOperations());

          // Replace all uses of the accesOp results and cache computation
          stencil::ReturnOp returnOp =
              cast<stencil::ReturnOp>(*std::prev(Block::iterator(accessOp)));
          replaceAccess(consumerOp, accessOp, producerResults,
                        returnOp.getOperands());
          offsetCache[offset] = returnOp.getOperands();
          rewriter.eraseOp(returnOp);
          clonedOp->erase();
        }
        rewriter.eraseOp(accessOp);
      }
    });

    // Update the all uses and copy the loop bounds
    for (size_t i = 0, e = consumerOp.getResults().size(); i != e; ++i)
      consumerOp.getResult(i).replaceAllUsesWith(newOp.getResult(i));

    // Erase the producer and consumer ops
    rewriter.eraseOp(consumerOp);
    rewriter.eraseOp(producerOp);
    return matchSuccess();
  }

  PatternMatchResult matchAndRewrite(stencil::ApplyOp applyOp,
                                     PatternRewriter &rewriter) const override {
    // Search producer apply op
    for (auto operand : applyOp.operands()) {
      if (auto producerOp =
              dyn_cast<stencil::ApplyOp>(operand->getDefiningOp())) {
        // Check only one consumer
        bool singleConsumer = true;
        for (auto result : producerOp.getResults()) {
          for (auto user : result.getUsers()) {
            if (isa<stencil::ApplyOp>(user) && user != applyOp.getOperation())
              singleConsumer = false;
            if (isa<stencil::StoreOp>(user))
              singleConsumer = false;
          }
        }
        // Ready to perform inlining
        if (singleConsumer)
          return inlineProducer(producerOp, applyOp, producerOp.getResults(),
                                rewriter);
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
  patterns.insert<ProducerInliningRewrite>(&getContext());
  applyPatternsGreedily(funcOp, patterns);
}

} // namespace

static PassRegistration<StencilInliningPass> pass("stencil-inlining",
                                                  "Inline stencil apply ops");
