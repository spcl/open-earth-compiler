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
#include "llvm/ADT/SmallVector.h"
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

  bool internalOnly;
};

// Fuse two apply ops connected to the same combine
struct FuseRewrite : public CombineLoweringPattern {
  using CombineLoweringPattern::CombineLoweringPattern;

  // Fuse two apply ops into a single apply op
  LogicalResult fuseApplyOps(stencil::ApplyOp applyOp1,
                             stencil::ApplyOp applyOp2,
                             stencil::CombineOp combineOp,
                             PatternRewriter &rewriter) const {
    // Check the shapes match
    auto shapeOp1 = cast<ShapeOp>(applyOp1.getOperation());
    auto shapeOp2 = cast<ShapeOp>(applyOp2.getOperation());
    if (shapeOp1.hasShape() && shapeOp2.hasShape() &&
        (shapeOp1.getLB() != shapeOp2.getLB() &&
         shapeOp1.getUB() != shapeOp2.getUB())) {
      combineOp.emitWarning("expected shapes to match");
      return failure();
    }

    // Compute the new result types
    SmallVector<Type, 10> newResultTypes;
    newResultTypes.append(applyOp1.getResultTypes().begin(),
                          applyOp1.getResultTypes().end());
    newResultTypes.append(applyOp2.getResultTypes().begin(),
                          applyOp2.getResultTypes().end());

    // Compute the new operands
    SmallVector<Value, 10> newOperands;
    newOperands.append(applyOp1.getOperands().begin(),
                       applyOp1.getOperands().end());
    newOperands.append(applyOp2.getOperands().begin(),
                       applyOp2.getOperands().end());

    // Get return operations
    auto returnOp1 =
        cast<stencil::ReturnOp>(applyOp1.getBody()->getTerminator());
    auto returnOp2 =
        cast<stencil::ReturnOp>(applyOp2.getBody()->getTerminator());

    // Check both apply operations have the same unroll configuration if any
    if (returnOp1.getUnrollFac() != returnOp2.getUnrollFac() ||
        returnOp1.getUnrollDim() != returnOp2.getUnrollDim()) {
      combineOp.emitWarning("expected matching unroll configurations");
      return failure();
    }

    // Introduce a new apply op
    auto newOp = rewriter.create<stencil::ApplyOp>(
        combineOp.getLoc(), newResultTypes, newOperands, applyOp1.lb(),
        applyOp1.ub());
    rewriter.mergeBlocks(
        applyOp1.getBody(), newOp.getBody(),
        newOp.getBody()->getArguments().take_front(applyOp1.getNumOperands()));
    rewriter.mergeBlocks(
        applyOp2.getBody(), newOp.getBody(),
        newOp.getBody()->getArguments().take_back(applyOp2.getNumOperands()));

    // Compute the new operands
    SmallVector<Value, 10> newReturnOperands;
    newReturnOperands.append(returnOp1.getOperands().begin(),
                             returnOp1.getOperands().end());
    newReturnOperands.append(returnOp2.getOperands().begin(),
                             returnOp2.getOperands().end());

    // Introduce a new return op
    rewriter.setInsertionPointToEnd(newOp.getBody());
    rewriter.create<stencil::ReturnOp>(combineOp.getLoc(), newReturnOperands,
                                       returnOp1.unroll());
    rewriter.eraseOp(returnOp1);
    rewriter.eraseOp(returnOp2);
    
    // Replace all uses of the two apply operations
    rewriter.replaceOp(
        applyOp1, newOp.getResults().take_front(applyOp1.getNumResults()));
    rewriter.replaceOp(
        applyOp2, newOp.getResults().take_back(applyOp2.getNumResults()));
    return success();
  }

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Handle the case if multiple applies are connected to lower
    auto definingLowerOps = combineOp.getLowerDefiningOps();
    if (definingLowerOps.size() > 1) {
      auto applyOp1 = cast<stencil::ApplyOp>(*definingLowerOps.begin());
      auto applyOp2 = cast<stencil::ApplyOp>(*(++definingLowerOps.begin()));
      return fuseApplyOps(applyOp1, applyOp2, combineOp, rewriter);
    }

    // Handle the case if multiple applyies are connected to higher
    auto definingUpperOps = combineOp.getUpperDefiningOps();
    if (definingUpperOps.size() > 1) {
      auto applyOp1 = cast<stencil::ApplyOp>(*definingUpperOps.begin());
      auto applyOp2 = cast<stencil::ApplyOp>(*(++definingUpperOps.begin()));
      return fuseApplyOps(applyOp1, applyOp2, combineOp, rewriter);
    }
    return failure();
  }
};

// Introduce empty stores to eliminate extra operands
struct EmptyStoreRewrite : public CombineLoweringPattern {
  using CombineLoweringPattern::CombineLoweringPattern;

  // Introduce empty stores for the extra operands
  stencil::ApplyOp addEmptyStores(stencil::ApplyOp applyOp, OperandRange range,
                                  stencil::CombineOp combineOp,
                                  PatternRewriter &rewriter) const {
    // Compute the result types
    SmallVector<Type, 10> newResultTypes(applyOp.getResultTypes().begin(),
                                         applyOp.getResultTypes().end());
    // Introduce the emtpy result types using the shape of the op
    auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
    for (auto operand : range) {
      auto newShape = applyFunElementWise(shapeOp.getUB(), shapeOp.getLB(),
                                          std::minus<int64_t>());
      newResultTypes.push_back(TempType::get(
          operand.getType().cast<TempType>().getElementType(), newShape));
    }

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
    for (auto operand : range) {
      auto resultOp = rewriter.create<stencil::StoreResultOp>(
          returnOp.getLoc(),
          ResultType::get(operand.getType().cast<TempType>().getElementType()),
          ValueRange());
      newOperands.append(returnOp.getUnrollFac(), resultOp);
    }
    rewriter.create<stencil::ReturnOp>(returnOp.getLoc(), newOperands,
                                       returnOp.unroll());
    rewriter.eraseOp(returnOp);
    return newOp;
  }

  // Iterate the operand range and append append the new op operands
  void appendOperandRange(stencil::ApplyOp oldOp, stencil::ApplyOp newOp,
                          OperandRange range,
                          SmallVector<Value, 10> &newOperands) const {
    for (auto value : range) {
      auto it = llvm::find(oldOp.getResults(), value);
      assert(it != oldOp.getResults().end() &&
             "expected to find the result matching the combine operand");
      newOperands.push_back(
          newOp.getResult(std::distance(oldOp.getResults().begin(), it)));
    }
  }

  // Reroute the store result of the apply ops via a combine op
  LogicalResult mirrorExtraResults(stencil::ApplyOp lowerOp,
                                   stencil::ApplyOp upperOp,
                                   stencil::CombineOp combineOp,
                                   PatternRewriter &rewriter) const {
    // Compute the updated apply operations
    auto newLowerOp =
        addEmptyStores(lowerOp, combineOp.upperext(), combineOp, rewriter);
    auto newUpperOp =
        addEmptyStores(upperOp, combineOp.lowerext(), combineOp, rewriter);

    // Update the combine operation
    SmallVector<Value, 10> newLowerOperands;
    SmallVector<Value, 10> newUpperOperands;
    // Append the lower and upper operands
    appendOperandRange(lowerOp, newLowerOp, combineOp.lower(),
                       newLowerOperands);
    appendOperandRange(upperOp, newUpperOp, combineOp.upper(),
                       newUpperOperands);
    // Append the extra operands of the lower and empty of the upper op
    appendOperandRange(lowerOp, newLowerOp, combineOp.lowerext(),
                       newLowerOperands);
    auto upperEmptyStores =
        newUpperOp.getResults().take_back(combineOp.lowerext().size());
    newUpperOperands.append(upperEmptyStores.begin(), upperEmptyStores.end());
    // Append the empty lower and the extra operands of the upper op
    auto lowerEmptyStores =
        newLowerOp.getResults().take_back(combineOp.upperext().size());
    newLowerOperands.append(lowerEmptyStores.begin(), lowerEmptyStores.end());
    appendOperandRange(upperOp, newUpperOp, combineOp.upperext(),
                       newUpperOperands);

    // Introduce a new stencil combine operation that has no extra operands
    rewriter.setInsertionPoint(combineOp);
    auto newOp = rewriter.create<stencil::CombineOp>(
        combineOp.getLoc(), combineOp.getResultTypes(), combineOp.dim(),
        combineOp.index(), newLowerOperands, newUpperOperands, ValueRange(),
        ValueRange(), combineOp.lbAttr(), combineOp.ubAttr());

    // Replace the combine operation
    rewriter.replaceOp(combineOp, newOp.getResults());
    rewriter.eraseOp(lowerOp);
    rewriter.eraseOp(upperOp);
    return success();
  }

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Handling extra operands is not needed
    if (combineOp.lowerext().empty() && combineOp.upperext().empty())
      return failure();

    // Handle multiple input operations first
    auto definingLowerOps = combineOp.getLowerDefiningOps();
    auto definingUpperOps = combineOp.getUpperDefiningOps();
    if (definingLowerOps.size() != 1 || definingUpperOps.size() != 1)
      return failure();

    // Try to get the lower and the upper apply op
    auto lowerOp = dyn_cast<stencil::ApplyOp>(*definingLowerOps.begin());
    auto upperOp = dyn_cast<stencil::ApplyOp>(*definingUpperOps.begin());
    if (lowerOp && upperOp) {
      return mirrorExtraResults(lowerOp, upperOp, combineOp, rewriter);
    }
    return failure();
  }
};

// TODO support multiple apply operations? -> separate pattern!

// Pattern replacing stencil.combine ops by if/else
struct IfElseRewrite : public CombineLoweringPattern {
  using CombineLoweringPattern::CombineLoweringPattern;

  // Apply the apply to combine op operand mapping to the return op operands
  SmallVector<Value, 10>
  permuteReturnOpOperands(stencil::ApplyOp applyOp,
                          OperandRange combineOpOperands,
                          stencil::ReturnOp returnOp) const {
    SmallVector<Value, 10> newOperands;
    // Compute a result to index mapping
    DenseMap<Value, unsigned> resultToIndex;
    for (auto result : applyOp.getResults()) {
      resultToIndex[result] = result.getResultNumber();
    }
    // Append the return op operands that correspond to the combine op operand
    for (auto value : combineOpOperands) {
      assert(value.getDefiningOp() == applyOp.getOperation() &&
             "expected operand is defined apply op");
      unsigned unrollFac = returnOp.getUnrollFac();
      auto returnOpOperands = returnOp.getOperands().slice(
          resultToIndex[value] * unrollFac, unrollFac);
      newOperands.append(returnOpOperands.begin(), returnOpOperands.end());
    }
    return newOperands;
  }

  // Lower the combine op to a if/else apply op
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
    if (lowerReturnOp.getUnrollFac() != upperReturnOp.getUnrollFac() ||
        lowerReturnOp.getUnrollDim() != upperReturnOp.getUnrollDim()) {
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
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
        lowerReturnOp,
        permuteReturnOpOperands(lowerOp, combineOp.lower(), lowerReturnOp));
    rewriter.setInsertionPoint(upperReturnOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
        upperReturnOp,
        permuteReturnOpOperands(upperOp, combineOp.upper(), upperReturnOp));

    // Move the computation to the new apply operation
    rewriter.mergeBlocks(
        lowerOp.getBody(), ifOp.getBody(0),
        newOp.getBody()->getArguments().take_front(lowerOp.getNumOperands()));
    rewriter.mergeBlocks(
        upperOp.getBody(), ifOp.getBody(1),
        newOp.getBody()->getArguments().take_front(upperOp.getNumOperands()));

    // Remove the combine op and the attached apply ops
    rewriter.replaceOp(combineOp, newOp.getResults());
    rewriter.eraseOp(upperOp);
    rewriter.eraseOp(lowerOp);
    return success();
  }

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {

    // Handle the extra operands first
    if (!combineOp.lowerext().empty() || !combineOp.upperext().empty())
      return failure();

    // Handle multiple input operations first
    auto definingLowerOps = combineOp.getLowerDefiningOps();
    auto definingUpperOps = combineOp.getUpperDefiningOps();
    if (definingLowerOps.size() != 1 || definingUpperOps.size() != 1)
      return failure();

    // Try to get the lower and the upper apply op
    auto lowerOp = dyn_cast<stencil::ApplyOp>(*definingLowerOps.begin());
    auto upperOp = dyn_cast<stencil::ApplyOp>(*definingUpperOps.begin());
    if (lowerOp && upperOp) {
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

  // TODO fix the entire domain size handling -> relax store / buffer verifiers!
  // // Check the combine tree root is connected to a buffer or store op
  // // (ensures the domain sizes of outputs written on a subdomain is known)
  // bool hasApplyOpUsers = false;
  // funcOp.walk([&](stencil::CombineOp combineOp) {
  //   if (llvm::any_of(combineOp.getOperation()->getUsers(), [](Operation *op)
  //   {
  //         return !(isa<stencil::BufferOp>(op) || isa<stencil::StoreOp>(op) ||
  //                  isa<stencil::CombineOp>(op));
  //       }))
  //     hasApplyOpUsers = true;
  // });
  // // Keep doing internal combine op lowerings by enhancing the output size
  // if (hasApplyOpUsers && !internalOnly) {
  //   funcOp.emitOpError("exectue shape correction before combine lowering");
  //   signalPassFailure();
  //   return;
  // }

  // TODO do not run all patterns on internal nodes?
  OwningRewritePatternList patterns;
  patterns.insert<IfElseRewrite, EmptyStoreRewrite, FuseRewrite>(&getContext(),
                                                                 internalOnly);
  applyPatternsAndFoldGreedily(funcOp, patterns);
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createStencilCombineLoweringPass() {
  return std::make_unique<StencilCombineLoweringPass>();
}
