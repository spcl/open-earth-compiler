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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <functional>
#include <iterator>
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

    // Replace the function by a function with an upadate signature
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

class AssertOpLowering : public StencilOpToStdPattern<stencil::AssertOp> {
public:
  using StencilOpToStdPattern<stencil::AssertOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto assertOp = cast<stencil::AssertOp>(operation);

    // Compute the static shape of the field and cast the input memref
    auto fieldType = assertOp.field().getType().cast<FieldType>();
    auto shape = computeShape(operation);
    auto resultType = typeConverter.convertFieldType(fieldType, shape);
    rewriter.create<MemRefCastOp>(loc, operands[0], resultType);
    rewriter.eraseOp(assertOp);
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

    // Get the assert and the cast operation
    auto castOp = getUserOp<MemRefCastOp>(operands[0]);
    assert(castOp && "exepected operands[0] to point to the input field");

    // Get the temp and field types
    auto fieldType = loadOp.field().getType().cast<FieldType>();
    auto tempType = loadOp.res().getType().cast<TempType>();

    // Compute the shape of the subviw
    auto subViewShape = computeSubViewShape(fieldType, operation,
                                            valueToLB.lookup(loadOp.field()));
    assert(std::get<1>(subViewShape) == tempType.getMemRefShape() &&
           "expected to get result memref shape");

    // Replace the load op by a subview op
    auto subViewOp = rewriter.create<SubViewOp>(
        loc, castOp.getResult(), std::get<0>(subViewShape),
        std::get<1>(subViewShape), std::get<2>(subViewShape), ValueRange(),
        ValueRange(), ValueRange());
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

    // Compute the loop bounds starting from zero
    // (in case of loop unrolling adjust the step of the loop)
    SmallVector<Value, 3> lb, ub, steps;
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    for (int64_t i = 0, e = shapeOp.getRank(); i != e; ++i) {
      lb.push_back(rewriter.create<ConstantIndexOp>(loc, shapeOp.getLB()[i]));
      ub.push_back(rewriter.create<ConstantIndexOp>(loc, shapeOp.getUB()[i]));
      steps.push_back(rewriter.create<ConstantIndexOp>(
          loc, returnOp.unroll().hasValue() ? returnOp.getUnroll()[i] : 1));
    }

    // Convert the signature of the apply op body
    // (access the apply op oparands directly and introduce the loop indicies)
    TypeConverter::SignatureConversion result(applyOp.getNumOperands());
    for (auto &en : llvm::enumerate(applyOp.getOperands())) {
      result.remapInput(en.index(), operands[en.index()]);
    }
    SmallVector<Type, 3> indexes(steps.size(),
                                 IndexType::get(applyOp.getContext()));
    result.addInputs(indexes);
    rewriter.applySignatureConversion(&applyOp.region(), result);

    // Replace the stencil apply operation by a parallel loop
    auto loop = rewriter.create<ParallelOp>(loc, lb, ub, steps);
    loop.getBody()->erase(); // TODO find better solution
    rewriter.inlineRegionBefore(applyOp.region(), loop.region(),
                                loop.region().begin());
    rewriter.setInsertionPointToEnd(loop.getBody());
    rewriter.create<YieldOp>(loc);
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
    if (!isa<ParallelOp>(operation->getParentOp()))
      return failure();
    auto loopOp = cast<ParallelOp>(operation->getParentOp());

    // Get allocations of result buffers
    SmallVector<Value, 10> allocVals;
    unsigned allocCount = returnOp.getNumOperands() / unrollFac;
    auto *node = loopOp.getOperation();
    while (node && allocVals.size() < allocCount) {
      if (auto allocOp = dyn_cast<AllocOp>(node))
        allocVals.insert(allocVals.begin(), allocOp.getResult());
      node = node->getPrevNode();
    }
    assert(allocVals.size() == allocCount &&
           "expected allocation for every result of the stencil operator");

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
        auto storeOffset = computeIndexValues(loopOp.getInductionVars(), offset,
                                              allocation, rewriter);
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

    // Get the parallel loop
    auto loopOp = operation->getParentOfType<ParallelOp>();
    if (!loopOp)
      return failure();
    assert(loopOp.getNumLoops() == accessOp.getOffset().size() &&
           "expected loop nest and access offset to have the same size");

    // Add the lower bound of the temporary to the access offset
    auto totalOffset =
        applyFunElementWise(accessOp.getOffset(), valueToLB[accessOp.temp()],
                            std::minus<int64_t>());
    auto tempType = accessOp.temp().getType().cast<TempType>();
    auto loadOffset = computeIndexValues(loopOp.getInductionVars(), totalOffset,
                                         tempType.getAllocation(), rewriter);

    // Replace the access op by a load op
    rewriter.replaceOpWithNewOp<mlir::LoadOp>(operation, operands[0],
                                              loadOffset);
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

    // Get the assert and the cast operation
    auto castOp = getUserOp<MemRefCastOp>(operands[1]);
    assert(castOp && "exepected operands[1] to point to the output field");

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
        loc, castOp.getResult(), std::get<0>(subViewShape),
        std::get<1>(subViewShape), std::get<2>(subViewShape), ValueRange(),
        ValueRange(), ValueRange());
    rewriter.replaceOp(allocOp, subViewOp.getResult());

    // Remove the deallocation and the store operation
    auto deallocOp = getUserOp<DeallocOp>(operands[0]);
    assert(deallocOp && "expecte dealloc operation");
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
  module.walk([&](stencil::AssertOp assertOp) {
    auto shapeOp = cast<ShapeOp>(assertOp.getOperation());
    valueToLB[assertOp.field()] = shapeOp.getLB();
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
  if (failed(applyFullConversion(module, target, patterns, &typeConverter))) {
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
      .insert<FuncOpLowering, AssertOpLowering, LoadOpLowering, ApplyOpLowering,
              ReturnOpLowering, AccessOpLowering, StoreOpLowering>(typeConveter,
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

Type StencilTypeConverter::convertFieldType(FieldType fieldType,
                                            ArrayRef<int64_t> shape) {
  Index revShape;
  for (auto en : llvm::enumerate(fieldType.getAllocation())) {
    // Insert at the front to convert from column to row-major
    if (en.value())
      revShape.insert(revShape.begin(), shape[en.index()]);
  }
  return MemRefType::get(revShape, fieldType.getElementType());
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

std::tuple<Index, Index, Index>
StencilToStdPattern::computeSubViewShape(FieldType fieldType, ShapeOp accessOp,
                                         Index assertLB) const {
  auto shape = computeShape(accessOp);
  Index revShape, revOffset, revStrides;
  for (auto en : llvm::enumerate(fieldType.getAllocation())) {
    // Insert values at the front to convert from column- to row-major
    if (en.value()) {
      revShape.insert(revShape.begin(), shape[en.index()]);
      revStrides.insert(revStrides.begin(), 1);
      revOffset.insert(revOffset.begin(),
                       accessOp.getLB()[en.index()] - assertLB[en.index()]);
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
      ValueRange params = {
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
