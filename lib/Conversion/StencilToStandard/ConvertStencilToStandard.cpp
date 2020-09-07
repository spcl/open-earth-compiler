#include "Conversion/StencilToStandard/ConvertStencilToStandard.h"
#include "Conversion/StencilToStandard/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <tuple>

using namespace mlir;
using namespace stencil;
using namespace scf;

namespace {

//===----------------------------------------------------------------------===//
// Rewriting Pattern
//===----------------------------------------------------------------------===//

class FuncOpLowering : public StencilOpToStdPattern<FuncOp> {
public:
  using StencilOpToStdPattern<FuncOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto funcOp = cast<FuncOp>(operation);

    // Convert the original function arguments
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    for (auto &en : llvm::enumerate(funcOp.getType().getInputs()))
      result.addInputs(en.index(), typeConverter.convertType(en.value()));
    auto funcType =
        FunctionType::get(result.getConvertedTypes(),
                          funcOp.getType().getResults(), funcOp.getContext());

    // Replace the function by a function with an updated signature
    auto newFuncOp =
        rewriter.create<FuncOp>(loc, funcOp.getName(), funcType, llvm::None);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Convert the signature and delete the original operation
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
    rewriter.eraseOp(funcOp);
    return success();
  }
};

class CastOpLowering : public StencilOpToStdPattern<stencil::CastOp> {
public:
  using StencilOpToStdPattern<stencil::CastOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto castOp = cast<stencil::CastOp>(operation);

    // Compute the static shape of the field and cast the input memref
    auto resType = castOp.res().getType().cast<FieldType>();
    rewriter.replaceOpWithNewOp<MemRefCastOp>(
        operation, operands[0],
        typeConverter.convertType(resType).cast<MemRefType>());
    return success();
  }
};

class LoadOpLowering : public StencilOpToStdPattern<stencil::LoadOp> {
public:
  using StencilOpToStdPattern<stencil::LoadOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto loadOp = cast<stencil::LoadOp>(operation);

    // Get the temp and field types
    auto fieldType = loadOp.field().getType().cast<FieldType>();
    auto tempType = loadOp.res().getType().cast<TempType>();

    // Compute the shape of the subview
    auto subViewShape = computeSubViewShape(fieldType, operation,
                                            valueToLB.lookup(loadOp.field()));
    assert(std::get<1>(subViewShape) == tempType.getMemRefShape() &&
           "expected to get result memref shape");

    // Replace the load op by a subview op
    auto subViewOp = rewriter.create<SubViewOp>(
        loc, operands[0], std::get<0>(subViewShape), std::get<1>(subViewShape),
        std::get<2>(subViewShape), ValueRange(), ValueRange(), ValueRange());
    rewriter.replaceOp(operation, subViewOp.getResult());
    return success();
  }
};

class ApplyOpLowering : public StencilOpToStdPattern<stencil::ApplyOp> {
public:
  using StencilOpToStdPattern<stencil::ApplyOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto applyOp = cast<stencil::ApplyOp>(operation);
    auto shapeOp = cast<ShapeOp>(operation);

    // Allocate storage for every stencil output
    SmallVector<Value, 10> newResults;
    for (unsigned i = 0, e = applyOp.getNumResults(); i != e; ++i) {
      assert(applyOp.getResult(i).getType().cast<TempType>().hasStaticShape() &&
             "expected the result types have a static shape");
      auto allocType = typeConverter.convertType(applyOp.getResult(i).getType())
                           .cast<MemRefType>();
      auto allocOp = rewriter.create<AllocOp>(loc, allocType);
      newResults.push_back(allocOp.getResult());
    }

    // Get the sequential dimension if there is one
    Value lb, ub, step;
    Optional<int64_t> sequential = None;
    if (applyOp.seq().hasValue()) {
      sequential = applyOp.getSeqDim();
      lb = rewriter.create<ConstantIndexOp>(loc, applyOp.getSeqLB());
      ub = rewriter.create<ConstantIndexOp>(loc, applyOp.getSeqUB());
      step = rewriter.create<ConstantIndexOp>(loc, 1);
    }

    // Compute the loop bounds starting from zero
    // (in case of loop unrolling adjust the step of the loop)
    SmallVector<Value, 3> lbs, ubs, steps;
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    for (int64_t i = 0, e = shapeOp.getRank(); i != e; ++i) {
      int64_t lb = shapeOp.getLB()[i];
      int64_t ub = sequential == i ? lb + 1 : shapeOp.getUB()[i];
      int64_t step = returnOp.unroll().hasValue() ? returnOp.getUnroll()[i] : 1;
      lbs.push_back(rewriter.create<ConstantIndexOp>(loc, lb));
      ubs.push_back(rewriter.create<ConstantIndexOp>(loc, ub));
      steps.push_back(rewriter.create<ConstantIndexOp>(loc, step));
      assert(sequential != i ||
             step == 1 && "expect sequential dimension to have step one");
    }

    // Convert the signature of the apply op body
    // (access the apply op operands and introduce the loop indicies)
    TypeConverter::SignatureConversion result(applyOp.getNumOperands());
    for (auto &en : llvm::enumerate(applyOp.getOperands())) {
      result.remapInput(en.index(), operands[en.index()]);
    }
    rewriter.applySignatureConversion(&applyOp.region(), result);

    // Affine map used for induction variable computation
    auto fwdExpr = rewriter.getAffineDimExpr(0);
    auto fwdMap = AffineMap::get(1, 0, fwdExpr);

    // Replace the stencil apply operation by a loop nest
    // (clone the innermost loop to remove the existing body)
    ParallelOp parallelOp = rewriter.create<ParallelOp>(loc, lbs, ubs, steps);
    // Add a sequential of parallel loop nest
    if (sequential) {
      rewriter.setInsertionPointToStart(parallelOp.getBody());
      auto forOp = rewriter.create<ForOp>(loc, lb, ub, step);
      rewriter.mergeBlockBefore(
          applyOp.getBody(),
          forOp.getLoopBody().getBlocks().back().getTerminator());

      // Insert index variables at the beginning of the loop body
      rewriter.setInsertionPointToStart(forOp.getBody());
      for (int64_t i = 0, e = shapeOp.getRank(); i != e; ++i) {
        if (sequential == i) {
          if (applyOp.getSeqDir() == 1) {
            // Access the iv of the sequential loop
            rewriter.create<AffineApplyOp>(loc, fwdMap,
                                           ValueRange(forOp.getInductionVar()));
          } else {
            // Reverse the iv of the sequential loop
            auto bwdExpr = rewriter.getAffineDimExpr(0) - 1 +
                           rewriter.getAffineDimExpr(1) -
                           rewriter.getAffineDimExpr(2);
            auto bwdMap = AffineMap::get(3, 0, bwdExpr);
            rewriter.create<AffineApplyOp>(
                loc, bwdMap, ValueRange({ub, lb, forOp.getInductionVar()}));
          }
          continue;
        }
        // Handle the parallel loop dimensions
        rewriter.create<AffineApplyOp>(
            loc, fwdMap, ValueRange(parallelOp.getInductionVars()[i]));
      }
    } else {
      rewriter.mergeBlockBefore(
          applyOp.getBody(),
          parallelOp.getLoopBody().getBlocks().back().getTerminator());

      // Insert index variables at the beginning of the loop body
      rewriter.setInsertionPointToStart(parallelOp.getBody());
      for (int64_t i = 0, e = shapeOp.getRank(); i != e; ++i) {
        rewriter.create<AffineApplyOp>(
            loc, fwdMap, ValueRange(parallelOp.getInductionVars()[i]));
      }
    }

    // Replace the applyOp
    rewriter.replaceOp(applyOp, newResults);

    // Deallocate the temporary storage
    rewriter.setInsertionPoint(
        applyOp.getParentRegion()->back().getTerminator());
    for (auto newResult : newResults) {
      rewriter.create<DeallocOp>(loc, newResult);
    }

    return success();
  }
};

class ReturnOpLowering : public StencilOpToStdPattern<stencil::ReturnOp> {
public:
  using StencilOpToStdPattern<stencil::ReturnOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto returnOp = cast<stencil::ReturnOp>(operation);

    /// Compute unroll factor and dimension
    auto unrollFac = returnOp.getUnrollFactor();
    size_t unrollDim = returnOp.getUnrollDimension();

    // Get the loop operation
    if (!isa<ParallelOp>(operation->getParentOp()) &&
        !isa<ForOp>(operation->getParentOp()))
      return failure();
    auto parallelOp = operation->getParentOfType<ParallelOp>();

    // Get allocations of result buffers
    SmallVector<Value, 10> allocVals;
    unsigned allocCount = returnOp.getNumOperands() / unrollFac;
    auto *node = parallelOp.getOperation();
    while (node && allocVals.size() < allocCount) {
      if (auto allocOp = dyn_cast<AllocOp>(node))
        allocVals.insert(allocVals.begin(), allocOp.getResult());
      node = node->getPrevNode();
    }
    assert(allocVals.size() == allocCount &&
           "expected allocation for every result of the stencil operator");

    // Get the induction variables
    auto inductionVars = getInductionVars(operation);

    // Introduce a store for every return value
    for (unsigned i = 0, e = allocCount; i != e; ++i) {
      for (unsigned j = 0, e = unrollFac; j != e; ++j) {
        rewriter.setInsertionPoint(returnOp);
        unsigned operandIdx = i * unrollFac + j;
        auto definingOp = returnOp.getOperand(operandIdx).getDefiningOp();
        if (definingOp && returnOp.getParentOp() == definingOp->getParentOp())
          rewriter.setInsertionPointAfter(definingOp);

        // Compute the store offset
        auto offset = valueToLB[returnOp.getOperand(operandIdx)];
        llvm::transform(offset, offset.begin(), std::negate<int64_t>());
        offset[unrollDim] += j;

        // Compute the index values and introduce the store operation
        SmallVector<bool, 3> allocation(offset.size(), true);
        auto storeOffset =
            computeIndexValues(inductionVars, offset, allocation, rewriter);
        rewriter.create<mlir::StoreOp>(loc, returnOp.getOperand(operandIdx),
                                       allocVals[i], storeOffset);
      }
    }

    rewriter.eraseOp(operation);
    return success();
  }
};

class AccessOpLowering : public StencilOpToStdPattern<stencil::AccessOp> {
public:
  using StencilOpToStdPattern<stencil::AccessOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto accessOp = cast<stencil::AccessOp>(operation);
    auto offsetOp = cast<OffsetOp>(accessOp.getOperation());

    // Get the induction variables
    auto inductionVars = getInductionVars(operation);
    if (inductionVars.size() == 0)
      return failure();
    assert(inductionVars.size() == offsetOp.getOffset().size() &&
           "expected loop nest and access offset to have the same size");

    // Add the lower bound of the temporary to the access offset
    auto totalOffset =
        applyFunElementWise(offsetOp.getOffset(), valueToLB[accessOp.temp()],
                            std::minus<int64_t>());
    auto tempType = accessOp.temp().getType().cast<TempType>();
    auto loadOffset = computeIndexValues(inductionVars, totalOffset,
                                         tempType.getAllocation(), rewriter);

    // Replace the access op by a load op
    rewriter.replaceOpWithNewOp<mlir::LoadOp>(operation, operands[0],
                                              loadOffset);
    return success();
  }
};

class DynAccessOpLowering : public StencilOpToStdPattern<stencil::DynAccessOp> {
public:
  using StencilOpToStdPattern<stencil::DynAccessOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto dynAccessOp = cast<stencil::DynAccessOp>(operation);

    // Get the induction variables
    auto inductionVars = getInductionVars(operation);
    if (inductionVars.size() == 0)
      return failure();
    assert(inductionVars.size() == dynAccessOp.offset().size() &&
           "expected loop nest and access offset to have the same size");

    // Add the negative lower bound to the offset
    auto tempType = dynAccessOp.temp().getType().cast<TempType>();
    auto tempLB = valueToLB[dynAccessOp.temp()];
    llvm::transform(tempLB, tempLB.begin(), std::negate<int64_t>());
    auto loadOffset = computeIndexValues(dynAccessOp.offset(), tempLB,
                                         tempType.getAllocation(), rewriter);

    // Replace the access op by a load op
    rewriter.replaceOpWithNewOp<mlir::LoadOp>(operation, operands[0],
                                              loadOffset);
    return success();
  }
};

class DependOpLowering : public StencilOpToStdPattern<stencil::DependOp> {
public:
  using StencilOpToStdPattern<stencil::DependOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto dependOp = cast<stencil::DependOp>(operation);
    auto offsetOp = cast<OffsetOp>(dependOp.getOperation());

    // Get the loop operation
    auto parallelOp = operation->getParentOfType<ParallelOp>();
    auto forOp = operation->getParentOfType<ForOp>();
    if (!parallelOp || !forOp)
      return failure();

    // Get the return operation
    auto returnOp = dyn_cast<stencil::ReturnOp>(
        forOp.getBody()->getTerminator()->getPrevNode());
    if (!returnOp)
      return failure();
    assert(returnOp.getUnrollFactor() == 1 &&
           "expect sequential loops to have no unrolling");

    // Get allocations of result buffers
    SmallVector<Value, 3> allocValues;
    auto *node = parallelOp.getOperation();
    while (node && allocValues.size() < returnOp.getNumOperands()) {
      if (auto allocOp = dyn_cast<AllocOp>(node)) {
        allocValues.insert(allocValues.begin(), allocOp.getResult());
      }
      node = node->getPrevNode();
    }
    assert(allocValues.size() > dependOp.output() &&
           "expected allocation for output index");

    // Get the induction variables
    auto inductionVars = getInductionVars(operation);

    // Add the lower bound of the result to the access offset
    auto totalOffset = applyFunElementWise(
        offsetOp.getOffset(), valueToLB[returnOp.getOperand(dependOp.output())],
        std::minus<int64_t>());
    SmallVector<bool, 3> allocation(totalOffset.size(), true);
    auto loadOffset =
        computeIndexValues(inductionVars, totalOffset, allocation, rewriter);

    // Replace the access op by a load op
    rewriter.replaceOpWithNewOp<mlir::LoadOp>(
        operation, allocValues[dependOp.output()], loadOffset);
    return success();
  }
};

class IndexOpLowering : public StencilOpToStdPattern<stencil::IndexOp> {
public:
  using StencilOpToStdPattern<stencil::IndexOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto indexOp = cast<stencil::IndexOp>(operation);
    auto offsetOp = cast<OffsetOp>(indexOp.getOperation());

    // Get the induction variables
    auto inductionVars = getInductionVars(operation);
    if (inductionVars.size() == 0)
      return failure();
    assert(inductionVars.size() == offsetOp.getOffset().size() &&
           "expected loop nest and access offset to have the same size");

    // Shift the induction variable by the offset
    auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
    auto map = AffineMap::get(2, 0, expr);
    SmallVector<Value, 2> params = {
        inductionVars[indexOp.dim()],
        rewriter
            .create<ConstantIndexOp>(loc, offsetOp.getOffset()[indexOp.dim()])
            .getResult()};

    // replace the index ob by an affine apply op
    rewriter.replaceOpWithNewOp<mlir::AffineApplyOp>(operation, map, params);

    return success();
  }
};

class StoreOpLowering : public StencilOpToStdPattern<stencil::StoreOp> {
public:
  using StencilOpToStdPattern<stencil::StoreOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto storeOp = cast<stencil::StoreOp>(operation);

    // Get the temp and field types
    auto fieldType = storeOp.field().getType().cast<FieldType>();
    auto tempType = storeOp.temp().getType().cast<TempType>();

    // Compute the shape of the subview
    auto subViewShape = computeSubViewShape(fieldType, operation,
                                            valueToLB.lookup(storeOp.field()));
    assert(std::get<1>(subViewShape) == tempType.getMemRefShape() &&
           "expected to get result memref shape");

    // Replace the allocation by a subview
    auto allocOp = operands[0].getDefiningOp();
    rewriter.setInsertionPoint(allocOp);
    auto subViewOp = rewriter.create<SubViewOp>(
        loc, operands[1], std::get<0>(subViewShape), std::get<1>(subViewShape),
        std::get<2>(subViewShape), ValueRange(), ValueRange(), ValueRange());
    rewriter.replaceOp(allocOp, subViewOp.getResult());

    // Remove the deallocation and the store operation
    auto deallocOp = getUserOp<DeallocOp>(operands[0]);
    assert(deallocOp && "expected dealloc operation");
    rewriter.eraseOp(deallocOp);
    rewriter.eraseOp(operation);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

class StencilToStdTarget : public ConversionTarget {
public:
  explicit StencilToStdTarget(MLIRContext &context)
      : ConversionTarget(context) {}

  bool isDynamicallyLegal(Operation *op) const override {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      return !funcOp.getAttr(
                 stencil::StencilDialect::getStencilProgramAttrName()) &&
             !funcOp.getAttr(
                 stencil::StencilDialect::getStencilFunctionAttrName());
    }
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Rewriting Pass
//===----------------------------------------------------------------------===//

struct StencilToStandardPass
    : public StencilToStandardPassBase<StencilToStandardPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }
  void runOnOperation() override;
};

void StencilToStandardPass::runOnOperation() {
  OwningRewritePatternList patterns;
  auto module = getOperation();

  // Check all shapes are set
  bool allShapesValid = true;
  module.walk([&](ShapeOp shapeOp) {
    if (!shapeOp.hasShape()) {
      allShapesValid = false;
      shapeOp.emitOpError("expected to have a valid shape");
      signalPassFailure();
    }
  });
  if (!allShapesValid)
    return;

  // Store the lower bounds of the input stencil program
  DenseMap<Value, Index> valueToLB;
  module.walk([&](stencil::CastOp castOp) {
    auto shapeOp = cast<ShapeOp>(castOp.getOperation());
    valueToLB[castOp.res()] = shapeOp.getLB();
  });
  module.walk([&](stencil::ApplyOp applyOp) {
    // Store the lower bounds for all arguments
    for (auto en : llvm::enumerate(applyOp.getOperands())) {
      if (auto shapeOp = dyn_cast_or_null<ShapeOp>(en.value().getDefiningOp()))
        valueToLB[applyOp.getBody()->getArgument(en.index())] = shapeOp.getLB();
    }
    // Store the lower bounds for all results
    auto LB = cast<ShapeOp>(applyOp.getOperation()).getLB();
    for (auto value : applyOp.getBody()->getTerminator()->getOperands()) {
      valueToLB[value] = LB;
    }
  });

  StencilTypeConverter typeConverter(module.getContext());
  populateStencilToStdConversionPatterns(typeConverter, valueToLB, patterns);

  StencilToStdTarget target(*(module.getContext()));
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<SCFDialect>();
  target.addDynamicallyLegalOp<FuncOp>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  if (failed(applyFullConversion(module, target, patterns))) {
    signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace stencil {

// Populate the conversion pattern list
void populateStencilToStdConversionPatterns(
    StencilTypeConverter &typeConveter, DenseMap<Value, Index> &valueToLB,
    mlir::OwningRewritePatternList &patterns) {
  patterns
      .insert<FuncOpLowering, CastOpLowering, LoadOpLowering, ApplyOpLowering,
              ReturnOpLowering, AccessOpLowering, DynAccessOpLowering,
              DependOpLowering, IndexOpLowering, StoreOpLowering>(typeConveter,
                                                                  valueToLB);
}

//===----------------------------------------------------------------------===//
// Stencil Type Converter
//===----------------------------------------------------------------------===//

StencilTypeConverter::StencilTypeConverter(MLIRContext *context_)
    : context(context_) {
  // Add a type conversion for the stencil field type
  addConversion([&](GridType type) {
    return MemRefType::get(type.getMemRefShape(), type.getElementType());
  });
  addConversion([&](Type type) -> Optional<Type> {
    if (auto gridType = type.dyn_cast<GridType>())
      return llvm::None;
    return type;
  });
}

//===----------------------------------------------------------------------===//
// Stencil Pattern Base Class
//===----------------------------------------------------------------------===//

StencilToStdPattern::StencilToStdPattern(StringRef rootOpName,
                                         StencilTypeConverter &typeConverter_,
                                         DenseMap<Value, Index> &valueToLB_,
                                         PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter_.getContext()),
      typeConverter(typeConverter_), valueToLB(valueToLB_) {}

Index StencilToStdPattern::computeShape(ShapeOp shapeOp) const {
  return applyFunElementWise(shapeOp.getUB(), shapeOp.getLB(),
                             std::minus<int64_t>());
}

SmallVector<Value, 3>
StencilToStdPattern::getInductionVars(Operation *operation) const {
  SmallVector<Value, 3> inductionVariables;

  // Get the parallel loop
  auto parallelOp = operation->getParentOfType<ParallelOp>();
  auto forOp = operation->getParentOfType<ForOp>();
  if (!parallelOp)
    return inductionVariables;

  // Collect the induction variables
  parallelOp.walk([&](AffineApplyOp applyOp) {
    for (auto operand : applyOp.getOperands()) {
      if (forOp && forOp.getInductionVar() == operand) {
        inductionVariables.push_back(applyOp.getResult());
        break;
      }
      if (llvm::is_contained(parallelOp.getInductionVars(), operand)) {
        inductionVariables.push_back(applyOp.getResult());
        break;
      }
    }
  });
  return inductionVariables;
}

std::tuple<Index, Index, Index>
StencilToStdPattern::computeSubViewShape(FieldType fieldType, ShapeOp accessOp,
                                         Index castLB) const {
  auto shape = computeShape(accessOp);
  Index revShape, revOffset, revStrides;
  for (auto en : llvm::enumerate(fieldType.getAllocation())) {
    // Insert values at the front to convert from column- to row-major
    if (en.value()) {
      revShape.insert(revShape.begin(), shape[en.index()]);
      revStrides.insert(revStrides.begin(), 1);
      revOffset.insert(revOffset.begin(),
                       accessOp.getLB()[en.index()] - castLB[en.index()]);
    }
  }
  return std::make_tuple(revOffset, revShape, revStrides);
}

SmallVector<Value, 3> StencilToStdPattern::computeIndexValues(
    ValueRange inductionVars, Index offset, ArrayRef<bool> allocation,
    ConversionPatternRewriter &rewriter) const {
  auto loc = rewriter.getInsertionPoint()->getLoc();
  auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
  auto map = AffineMap::get(2, 0, expr);
  SmallVector<Value, 3> resOffset;
  for (auto en : llvm::enumerate(allocation)) {
    // Insert values at the front to convert from column- to row-major
    if (en.value()) {
      SmallVector<Value, 2> params = {
          inductionVars[en.index()],
          rewriter.create<ConstantIndexOp>(loc, offset[en.index()])
              .getResult()};
      auto affineApplyOp = rewriter.create<AffineApplyOp>(loc, map, params);
      resOffset.insert(resOffset.begin(), affineApplyOp.getResult());
    }
  }
  return resOffset;
}

} // namespace stencil
} // namespace mlir

std::unique_ptr<Pass> mlir::createConvertStencilToStandardPass() {
  return std::make_unique<StencilToStandardPass>();
}
