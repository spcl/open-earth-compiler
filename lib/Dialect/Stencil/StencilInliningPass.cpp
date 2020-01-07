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

struct InliningRewrite : public OpRewritePattern<stencil::ApplyOp> {
  using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

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
    // Compute the operand list for the fused apply op and the index mapping
    SmallVector<Value, 10> newOperands;
    SmallVector<int, 10> consumerIdx(consumerOp.operands().size());
    SmallVector<int, 10> producerIdx(producerOp.operands().size());
    llvm::transform(consumerOp.operands(), consumerIdx.begin(), [&](Value x) {
      if (!llvm::is_contained(producerResults, x)) {
        newOperands.push_back(x);
        return static_cast<int>(newOperands.size()) - 1;
      }
      return -1;
    });
    llvm::transform(producerOp.operands(), producerIdx.begin(), [&](Value x) {
      auto iter = llvm::find(newOperands, x);
      if (iter == newOperands.end()) {
        newOperands.push_back(x);
        return static_cast<int>(newOperands.size()) - 1;
      }
      return static_cast<int>(std::distance(newOperands.begin(), iter));
    });

    // Clone the consumer op
    auto loc = consumerOp.getLoc();
    auto newOp = rewriter.create<stencil::ApplyOp>(loc, newOperands,
                                                   consumerOp.getResults());
    rewriter.eraseOp(newOp.getBody()->getTerminator());

    // Compute the value mapping
    BlockAndValueMapping mapper;
    for (size_t i = 0, e = consumerIdx.size(); i != e; ++i) {
      if (consumerIdx[i] != -1)
        mapper.map(consumerOp.getBody()->getArgument(i),
                   newOp.getBody()->getArgument(consumerIdx[i]));
    }
    for (size_t i = 0, e = producerIdx.size(); i != e; ++i) {
      assert(producerIdx[i] != -1 &&
             "expected producer arguments to have a valid index");
      mapper.map(producerOp.getBody()->getArgument(i),
                 newOp.getBody()->getArgument(producerIdx[i]));
    }

    // Clone the consumer op and store the producer accesses
    SmallVector<stencil::AccessOp, 10> producerAccessOps;
    rewriter.setInsertionPointToStart(newOp.getBody());
    for (Operation &op : consumerOp.getBody()->getOperations()) {
      auto clonedOp = rewriter.clone(op, mapper);
      // Inline all producer accesses
      if (auto accessOp = dyn_cast<stencil::AccessOp>(clonedOp)) {
        if (llvm::is_contained(consumerOp.getBody()->getArguments(),
                               clonedOp->getOpOperand(0).get())) {
          producerAccessOps.push_back(accessOp);
        }
      }
    }

    // Replace all producer accesses
    DenseMap<ArrayRef<int64_t>, ValueRange> offsetCache;
    for (auto producerAccessOp : producerAccessOps) {
      // Clone the producer ops
      ArrayRef<int64_t> offset = producerAccessOp.getOffset();
      // Clone the producer into the consumer
      if (offsetCache.count(offset) != 0) {
        // Used cached values if possible
        replaceAccess(consumerOp, producerAccessOp, producerResults,
                      offsetCache[offset]);
      } else {
        // Otherwise clone all operations
        rewriter.setInsertionPoint(producerAccessOp);
        for (Operation &op : producerOp.getBody()->getOperations()) {
          auto clonedOp = rewriter.clone(op, mapper);
          // Shift all access ops
          if (auto accessOp = dyn_cast<stencil::AccessOp>(clonedOp)) {
            SmallVector<int64_t, 3> sum(offset.size());
            llvm::transform(llvm::zip(offset, accessOp.getOffset()),
                            sum.begin(), [](std::tuple<int64_t, int64_t> x) {
                              return std::get<0>(x) + std::get<1>(x);
                            });
            accessOp.setOffset(sum);
          }
          // Replaces uses of accessOp with returnOp operand
          if (auto returnOp = dyn_cast<stencil::ReturnOp>(clonedOp)) {
            replaceAccess(consumerOp, producerAccessOp, producerResults,
                          returnOp.getOperands());
            offsetCache[offset] = returnOp.getOperands();
            rewriter.eraseOp(returnOp);
          }
        }
      }
      rewriter.eraseOp(producerAccessOp);
    }

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
  patterns.insert<InliningRewrite>(&getContext());
  applyPatternsGreedily(funcOp, patterns);
}

} // namespace

static PassRegistration<StencilInliningPass> pass("stencil-inlining",
                                                  "Inline stencil apply ops");
