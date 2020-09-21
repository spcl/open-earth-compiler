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
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <iterator>

using namespace mlir;
using namespace stencil;

namespace {

// Base class of the combine lowering patterns
struct CombineLoweringPattern : public OpRewritePattern<stencil::CombineOp> {
  CombineLoweringPattern(MLIRContext *context, bool internalOnly,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<stencil::CombineOp>(context, benefit),
        internalOnly(internalOnly) {}

  bool isConnectedToStoreOp(stencil::ApplyOp applyOp) const {
    return llvm::any_of(applyOp.getOperation()->getUsers(), [](Operation *op) {
      return isa<stencil::StoreOp>(op);
    });
  }

  bool internalOnly;
};

// Reroute apply op outputs through the combine op
struct RerouteRewrite : public CombineLoweringPattern {
  using CombineLoweringPattern::CombineLoweringPattern;

  // Compute the list of stores results
  SmallVector<OpResult, 10> getStoreResults(stencil::ApplyOp applyOp) const {
    SmallVector<OpResult, 10> storeResults;
    llvm::copy_if(applyOp.getResults(), std::back_inserter(storeResults),
                  [](OpResult result) {
                    return llvm::any_of(result.getUsers(), [](Operation *op) {
                      return isa<stencil::ReturnOp>(op);
                    });
                  });
    assert(llvm::all_of(storeResults,
                        [](OpResult result) { return result.hasOneUse(); }) &&
           "expected store results to have one use");
    return storeResults;
  }

  // Introduce empty stores for the store results of the neighbor
  stencil::ApplyOp addEmptyStores(stencil::ApplyOp applyOp,
                                  ArrayRef<OpResult> storeResults,
                                  stencil::CombineOp combineOp,
                                  PatternRewriter &rewriter) const {
    // Compute the result types
    SmallVector<Type, 10> newResultTypes(applyOp.getResultTypes().begin(),
                                         applyOp.getResultTypes().end());
    llvm::transform(
        storeResults, std::back_inserter(newResultTypes), [&](Value value) {
          auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
          auto storeType = value.getType().cast<TempType>();
          auto storeShape = storeType.getShape();
          auto newShape = applyFunElementWise(shapeOp.getUB(), shapeOp.getLB(),
                                              std::minus<int64_t>());
          // Check that the shapes match except for the combine dimension
          assert(llvm::all_of(llvm::enumerate(newShape),
                              [&](auto en) {
                                return en.value() == storeShape[en.index()] ||
                                       en.index() == combineOp.dim();
                              }) &&
                 "expected shapes of the upper and lower apply to match");
          return TempType::get(storeType.getElementType(), newShape);
        });

    // Replace the apply operation
    rewriter.setInsertionPoint(applyOp);
    auto newOp = rewriter.create<stencil::ApplyOp>(
        applyOp.getLoc(), newResultTypes, applyOp.getOperands(), applyOp.lb(),
        applyOp.ub());
    rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(),
                         newOp.getBody()->getArguments());

    // Get the return operation
    auto returnOp = cast<stencil::ReturnOp>(newOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(returnOp);

    // Insert the empty stores
    SmallVector<Value, 10> newOperands = returnOp.getOperands();
    for (auto storeResult : storeResults) {
      auto elemType = storeResult.getType().cast<TempType>().getElementType();
      auto resultOp = rewriter.create<stencil::StoreResultOp>(
          returnOp.getLoc(), elemType, ValueRange());
      newOperands.append(returnOp.getUnrollFactor(), resultOp);
    }
    rewriter.create<stencil::ReturnOp>(returnOp.getLoc(), newOperands,
                                       returnOp.unroll());
    rewriter.eraseOp(returnOp);
    return newOp;
  }

  void appendCombineResults(stencil::ApplyOp applyOp, stencil::ApplyOp newOp,
                            stencil::CombineOp combineOp,
                            SmallVector<Value, 10> &newOperands) const {
    for (OpResult result : applyOp.getResults()) {
      if (llvm::is_contained(result.getUsers(), combineOp.getOperation())) {
        assert(result.hasOneUse() && "expected the result to have one use");
        newOperands.push_back(newOp.getResult(result.getResultNumber()));
      }
    }
  }

  void appendStoreResults(stencil::ApplyOp newOp,
                          ArrayRef<OpResult> storeResults,
                          SmallVector<Value, 10> &newOperands) const {
    for (OpResult result : storeResults) {
      newOperands.push_back(newOp.getResult(result.getResultNumber()));
    }
  }

  void appendEmptyResults(stencil::ApplyOp newOp,
                          ArrayRef<OpResult> storeResults,
                          SmallVector<Value, 10> &newOperands) const {
    auto emptyStores = newOp.getResults().take_back(storeResults.size());
    newOperands.append(emptyStores.begin(), emptyStores.end());
  }

  LogicalResult rerouteStoreResults(stencil::ApplyOp lowerOp,
                                    stencil::ApplyOp upperOp,
                                    ArrayRef<OpResult> lowerStoreResults,
                                    ArrayRef<OpResult> upperStoreResults,
                                    stencil::CombineOp combineOp,
                                    PatternRewriter &rewriter) const {
    // Compute the updated apply operations
    auto newLowerOp =
        addEmptyStores(lowerOp, upperStoreResults, combineOp, rewriter);
    auto newUpperOp =
        addEmptyStores(upperOp, lowerStoreResults, combineOp, rewriter);

    // Compute the new result types
    SmallVector<Type, 10> newResultTypes(combineOp.getResultTypes().begin(),
                                         combineOp.getResultTypes().end());
    llvm::transform(lowerStoreResults, std::back_inserter(newResultTypes),
                    [](Value result) { return result.getType(); });
    llvm::transform(upperStoreResults, std::back_inserter(newResultTypes),
                    [](Value result) { return result.getType(); });

    // Update the combine operation
    SmallVector<Value, 10> newLowerOperands;
    SmallVector<Value, 10> newUpperOperands;
    appendCombineResults(lowerOp, newLowerOp, combineOp, newLowerOperands);
    appendCombineResults(upperOp, newUpperOp, combineOp, newUpperOperands);
    // Append the stores of the lower apply op
    appendStoreResults(newLowerOp, lowerStoreResults, newLowerOperands);
    appendEmptyResults(newUpperOp, lowerStoreResults, newUpperOperands);
    // Append the stores of the upper apply op
    appendEmptyResults(newLowerOp, upperStoreResults, newLowerOperands);
    appendStoreResults(newUpperOp, upperStoreResults, newUpperOperands);

    // Introduce a new stencil apply right after the later apply op
    rewriter.setInsertionPointAfter(
        lowerOp.getOperation()->isBeforeInBlock(upperOp.getOperation())
            ? upperOp
            : lowerOp);
    auto newOp = rewriter.create<stencil::CombineOp>(
        combineOp.getLoc(), newResultTypes, combineOp.dim(), combineOp.index(),
        newLowerOperands, newUpperOperands, combineOp.lbAttr(),
        combineOp.ubAttr());

    // Replace the combine operation
    auto repResults = newOp.getResults().take_front(combineOp.getNumResults());
    rewriter.replaceOp(combineOp, repResults);

    // Replace the store results
    auto lowerRepResults =
        newLowerOp.getResults().take_front(lowerOp.getNumResults());
    auto upperRepResults =
        newUpperOp.getResults().take_front(upperOp.getNumResults());
    rewriter.replaceOp(lowerOp, lowerRepResults);
    rewriter.replaceOp(upperOp, upperRepResults);
    return success();
  }

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Get the lower and the upper apply up
    auto lowerOp = dyn_cast<stencil::ApplyOp>(combineOp.getLowerOp());
    auto upperOp = dyn_cast<stencil::ApplyOp>(combineOp.getUpperOp());
    if (lowerOp && upperOp) {
      auto lowerStoreResults = getStoreResults(lowerOp);
      auto upperStoreResults = getStoreResults(upperOp);
      if (lowerStoreResults.size() > 0 || upperStoreResults.size() > 0) {
        return rerouteStoreResults(upperOp, lowerOp, upperStoreResults,
                                   lowerStoreResults, combineOp, rewriter);
      }
    }
    return failure();
  }
};

// Pattern replacing stencil.combine ops by if/else
struct IfElseRewrite : public CombineLoweringPattern {
  using CombineLoweringPattern::CombineLoweringPattern;

  LogicalResult lowerStencilCombine(stencil::ApplyOp lowerOp,
                                    stencil::ApplyOp upperOp,
                                    stencil::CombineOp combineOp,
                                    PatternRewriter &rewriter) const {
    auto loc = combineOp.getLoc();
    auto shapeOp = cast<stencil::ShapeOp>(combineOp.getOperation());

    // Compute the operands of the fused apply op
    // (run canonicalization after the pass to cleanup arguments)
    SmallVector<Value, 10> newOperands = lowerOp.getOperands();
    newOperands.append(upperOp.getOperands().begin(),
                       upperOp.getOperands().end());

    // Create a new apply op that updates the lower and upper domains
    // (rerun shape inference after the pass to avoid bound computations)
    auto newOp = rewriter.create<stencil::ApplyOp>(
        loc, combineOp.getResultTypes(), newOperands, combineOp.lb(),
        combineOp.ub());
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
      if (isConnectedToStoreOp(lowerOp) || isConnectedToStoreOp(upperOp))
        return failure();

      // Check the apply op is an internal op if the pass flag is set
      if (internalOnly) {
        auto rootOp = combineOp.getCombineTreeRoot().getOperation();
        if (llvm::none_of(rootOp->getUsers(), [](Operation *op) {
              return isa<stencil::ApplyOp>(op);
            }))
          return failure();
      }

      // Lower the combine op and its predecessors to a single apply
      return lowerStencilCombine(lowerOp, upperOp, combineOp, rewriter);
    }
    return failure();
  }
}; // namespace

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
  patterns.insert<IfElseRewrite, RerouteRewrite>(&getContext(), internalOnly);
  applyPatternsAndFoldGreedily(funcOp, patterns);
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createStencilCombineLoweringPass() {
  return std::make_unique<StencilCombineLoweringPass>();
}
