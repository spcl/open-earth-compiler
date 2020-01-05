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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>

using namespace mlir;

namespace {

struct InliningRewrite : public OpRewritePattern<stencil::ApplyOp> {
  using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

  PatternMatchResult inlineProducer(stencil::ApplyOp producerOp,
                                    stencil::ApplyOp consumerOp, Value edge,
                                    PatternRewriter &rewriter) const {
    // Computing of producer and consumer arguments to replacement
    SmallVector<Value, 10> replacementOperands;
    DenseMap<unsigned, unsigned> consumerArgMapping;
    DenseMap<unsigned, unsigned> producerArgMapping;
    unsigned idx = consumerOp.operands().size();
    for (unsigned i = 0, e = consumerOp.operands().size(); i != e; ++i) {
      if (consumerOp.getOperand(i) == edge)
        continue;
      replacementOperands.push_back(consumerOp.getOperand(i));
      consumerArgMapping[i] = idx;
      ++idx;
    }
    for (unsigned i = 0, e = producerOp.operands().size(); i != e; ++i) {
      auto iter = llvm::find(replacementOperands, producerOp.getOperand(i));
      if (iter == replacementOperands.end()) {
        replacementOperands.push_back(producerOp.getOperand(i));
        producerArgMapping[i] = idx;
        ++idx;
      } else {
        producerArgMapping[i] =
            std::distance(replacementOperands.begin(), iter);
      }
    }

    // Clone the consumer op
    auto loc = consumerOp.getLoc();
    auto replacementOp = rewriter.create<stencil::ApplyOp>(
        loc, replacementOperands, consumerOp.getResults());

    BlockAndValueMapping mapping;
    consumerOp.region().cloneInto(&replacementOp.region(), mapping);

    // Add updated argument list at the end
    for (auto operand : replacementOperands)
      replacementOp.getBody()->addArgument(operand.getType());

    // Replace all argument uses with new arguments
    for (unsigned i = 0, e = consumerOp.operands().size(); i != e; ++i) {
      if (consumerArgMapping.count(i) != 0) {
        unsigned replacementIdx = consumerArgMapping[i];
        replacementOp.getBody()->getArgument(i).replaceAllUsesWith(
            replacementOp.getBody()->getArgument(replacementIdx));
      } else {
        // Replace all uses of consumer op
        for (auto &use : replacementOp.getBody()->getArgument(i).getUses()) {
          auto accessOp = dyn_cast<stencil::AccessOp>(use.getOwner());
          ArrayRef<int64_t> offset = accessOp.getOffset();

          // Clone the producer op
          BlockAndValueMapping mapper;
          for (unsigned i = 0, e = producerOp.getBody()->getNumArguments();
               i != e; ++i)
            mapper.map(
                producerOp.getBody()->getArgument(i),
                replacementOp.getBody()->getArgument(producerArgMapping[i]));
          
          rewriter.setInsertionPointToStart(replacementOp.getBody());
          for (Operation &op : producerOp.getBody()->getOperations()) {
            auto clone = rewriter.clone(op, mapper);
            if(auto accessOp = dyn_cast<stencil::AccessOp>(clone)) {
              SmallVector<int64_t, 3> sum(offset.size());
              llvm::transform(llvm::zip(offset, accessOp.getOffset()),
                              sum.begin(), [](std::tuple<int64_t, int64_t> x) {
                                return std::get<0>(x) + std::get<1>(x);
                              });
              accessOp.setOffset(sum);
            }
            if(auto returnOp = dyn_cast<stencil::ReturnOp>(clone)) {
              accessOp.getResult()->replaceAllUsesWith(returnOp.getOperand(0));
              rewriter.eraseOp(returnOp);
              break;
            }
          }
          rewriter.eraseOp(accessOp);
        }
      }
    }

    // Remove the original arguments of the consumer op
    for (unsigned i = 0, e = consumerOp.operands().size(); i != e; ++i)
      replacementOp.getBody()->eraseArgument(0);

    // Update the all uses and copy the loop bounds
    for (size_t i = 0, e = consumerOp.getResults().size(); i != e; ++i) 
      consumerOp.getResult(i).replaceAllUsesWith(replacementOp.getResult(i));
    
    // Erase the producer and consumer ops
    // rewriter.eraseOp(producerOp);
    // rewriter.eraseOp(consumerOp);
    
    return matchSuccess();
  }

  PatternMatchResult matchAndRewrite(stencil::ApplyOp applyOp,
                                     PatternRewriter &rewriter) const override {
    // Search producer apply op
    for (auto operand : applyOp.operands()) {
      if (auto producerOp =
              dyn_cast<stencil::ApplyOp>(operand->getDefiningOp())) {
        // Check only one consumer
        bool multipleConsumers = false;
        for (auto &use : operand.getUses()) {
          if (auto consumerOp = dyn_cast<stencil::ApplyOp>(use.getOwner()))
            if (consumerOp != applyOp)
              multipleConsumers = true;
          if (isa<stencil::StoreOp>(use.getOwner()))
            multipleConsumers = true;
        }
        // Try the next operand
        if (multipleConsumers)
          continue;
        if (producerOp.getNumResults() != 1)
          continue;
        // Ready to perform inlining
        return inlineProducer(producerOp, applyOp, operand, rewriter);
      }
    }
    return matchFailure();
  }
};

struct StencilInliningPass : public FunctionPass<StencilInliningPass> {
  void runOnFunction() override;
};

void StencilInliningPass::runOnFunction() {
  OwningRewritePatternList patterns;
  FuncOp funcOp = getFunction();
  auto *context = &getContext();
  patterns.insert<InliningRewrite>(context);
  applyPatternsGreedily(funcOp, patterns);
}

} // namespace

static PassRegistration<StencilInliningPass> pass("stencil-inlining",
                                                  "Inline stencil apply ops");
