#include "Conversion/StencilToStandard/ConvertStencilToStandard.h"
#include "Conversion/StencilToStandard/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
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
#include <iterator>

using namespace mlir;
using namespace stencil;

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
    auto shape = computeOpShape(operation, fieldType.getAllocation());
    assert(shape.hasValue() && "expected assertOp to have a shape");
    auto resultType =
        MemRefType::get(shape.getValue(), fieldType.getElementType());
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
    auto assertOp = getUserOp<stencil::AssertOp>(loadOp.field());
    if (!castOp || !assertOp)
      return failure();

    // Compute the memref type that stores the loaded data
    auto resultType = computeResultType(
        castOp.getResult().getType().cast<MemRefType>(),
        loadOp.res().getType().cast<TempType>().getAllocation(),
        assertOp.getOperation(), operation);
    if (!resultType.hasValue())
      return failure();

    // Replace the load operation by a subview operation
    auto subViewOp = rewriter.create<SubViewOp>(loc, resultType.getValue(),
                                                castOp.getResult());
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

    // Allocate storage for every stencil output
    SmallVector<Value, 10> newResults;
    for (unsigned i = 0, e = applyOp.getNumResults(); i != e; ++i) {
      TempType tempType = applyOp.getResult(i).getType().cast<TempType>();
      auto strides = computeStrides(tempType.getMemRefShape());
      auto affineMap =
          makeStridedLinearLayoutMap(strides, 0, applyOp.getContext());
      auto allocType = MemRefType::get(tempType.getMemRefShape(),
                                       tempType.getElementType(), affineMap, 0);

      auto allocOp = rewriter.create<AllocOp>(loc, allocType);
      newResults.push_back(allocOp.getResult());
    }

    // Compute the shape of the operation
    auto shape = computeOpShape(
        operation,
        applyOp.getResults()[0].getType().cast<TempType>().getAllocation());
    if (!shape.hasValue())
      return failure();

    // Compute the loop bounds starting from zero
    // (in case of loop unrolling adjust the step of the loop)
    SmallVector<Value, 3> lb;
    SmallVector<Value, 3> ub;
    SmallVector<Value, 3> steps;
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    for (int64_t i = 0, e = shape.getValue().size(); i != e; ++i) {
      lb.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
      ub.push_back(rewriter.create<ConstantIndexOp>(loc, shape.getValue()[i]));
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
    auto loop = rewriter.create<loop::ParallelOp>(loc, lb, ub, steps);
    loop.getBody()->erase(); // TODO find better solution
    rewriter.inlineRegionBefore(applyOp.region(), loop.region(),
                                loop.region().begin());
    rewriter.setInsertionPointToEnd(loop.getBody());
    rewriter.create<loop::YieldOp>(loc);
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

    // Get the parallel loop
    if (!isa<loop::ParallelOp>(operation->getParentOp()))
      return failure();
    auto loop = cast<loop::ParallelOp>(operation->getParentOp());
    SmallVector<Value, 3> loopIVs(loop.getNumLoops());
    llvm::transform(loop.getInductionVars(), loopIVs.begin(),
                    [](Value blockArg) { return blockArg; });

    // Get temporary buffers
    SmallVector<Operation *, 10> allocOps;
    Operation *currentOp = loop.getOperation();
    // Skip the loop constants
    while (isa<ConstantOp>(currentOp->getPrevNode())) {
      currentOp = currentOp->getPrevNode();
      assert(currentOp && "failed to find allocation for results");
    }
    // Compute the number of result temps
    unsigned numResults =
        returnOp.getNumOperands() / returnOp.getUnrollFactor();
    // assert(returnOp.getNumOperands() % returnOp.getUnrollFactor() == 0 &&
    //        "expected number of operands to be a multiple of the unroll
    //        factor");
    for (unsigned i = 0, e = numResults; i != e; ++i) {
      currentOp = currentOp->getPrevNode();
      allocOps.push_back(currentOp);
      assert(dyn_cast<AllocOp>(currentOp) &&
             "failed to find allocation for results");
    }
    SmallVector<Value, 10> allocVals(allocOps.size());
    llvm::transform(llvm::reverse(allocOps), allocVals.begin(),
                    [](Operation *allocOp) { return allocOp->getResult(0); });

    // Replace the return op by store ops
    auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
    auto map = AffineMap::get(2, 0, expr);
    for (unsigned i = 0, e = numResults; i != e; ++i) {
      for (unsigned j = 0, e = returnOp.getUnrollFactor(); j != e; ++j) {
        rewriter.setInsertionPoint(returnOp);
        unsigned operandIdx = i * returnOp.getUnrollFactor() + j;
        auto definingOp = returnOp.getOperand(operandIdx).getDefiningOp();
        if (definingOp && returnOp.getParentOp() == definingOp->getParentOp())
          rewriter.setInsertionPointAfter(definingOp);
        // Add the unrolling offset to the loop ivs
        SmallVector<Value, 3> storeOffset = loopIVs;
        if (j > 0) {
          auto unroll = returnOp.getUnroll();
          assert(llvm::count(unroll, 1) == unroll.size() - 1 &&
                 "expected a single non-zero entry");
          auto it = llvm::find_if(unroll, [&](int64_t x) {
            return x == returnOp.getUnrollFactor();
          });
          auto unrollDim = std::distance(unroll.begin(), it);
          auto constantOp = rewriter.create<ConstantIndexOp>(loc, j);
          ValueRange params = {loopIVs[unrollDim], constantOp.getResult()};
          auto affineApplyOp = rewriter.create<AffineApplyOp>(loc, map, params);
          storeOffset[unrollDim] = affineApplyOp.getResult();
        }
        rewriter.create<mlir::StoreOp>(loc, returnOp.getOperand(i),
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
    auto loopOp = operation->getParentOfType<loop::ParallelOp>();
    if (!loopOp)
      return failure();
    assert(loopOp.getNumLoops() == accessOp.getOffset().size() &&
           "expected loop nest and access offset to have the same size");
    SmallVector<Value, 3> loopIVs(loopOp.getNumLoops());
    llvm::transform(loopOp.getInductionVars(), loopIVs.begin(),
                    [](Value arg) { return arg; });

    // Compute the access offsets
    auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
    auto map = AffineMap::get(2, 0, expr);
    auto offset = accessOp.getOffset();
    SmallVector<Value, 3> loadOffset;
    for (size_t i = 0, e = offset.size(); i != e; ++i) {
      auto constantOp = rewriter.create<ConstantIndexOp>(loc, offset[i]);
      ValueRange params = {loopIVs[i], constantOp.getResult()};
      auto affineApplyOp = rewriter.create<AffineApplyOp>(loc, map, params);
      loadOffset.push_back(affineApplyOp.getResult());
    }
    // assert(loadOffset.size() ==
    //            operands[0].getType().cast<MemRefType>().getRank() &&
    //        "expected load offset size to match memref rank");

    // accessOp.temp().getType().dump();
    // operands[0].getType().dump();

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
    auto assertOp = getUserOp<stencil::AssertOp>(storeOp.field());
    if (!castOp || !assertOp)
      return failure();

    // Compute the memref type that stores the loaded data
    auto resultType = computeResultType(
        castOp.getResult().getType().cast<MemRefType>(),
        storeOp.temp().getType().cast<TempType>().getAllocation(),
        assertOp.getOperation(), operation);
    if (!resultType.hasValue())
      return failure();

    // Remove allocation and deallocation and insert subtemp op
    auto allocOp = operands[0].getDefiningOp();
    rewriter.setInsertionPoint(allocOp);
    auto subViewOp = rewriter.create<SubViewOp>(loc, resultType.getValue(),
                                                castOp.getResult());
    rewriter.replaceOp(allocOp, subViewOp.getResult());

    // TODO remove dealloc

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

  StencilTypeConverter typeConverter(module.getContext());
  populateStencilToStdConversionPatterns(typeConverter, patterns);

  StencilToStdTarget target(*(module.getContext()));
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<loop::LoopOpsDialect>();
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
    StencilTypeConverter &typeConveter,
    mlir::OwningRewritePatternList &patterns) {
  patterns
      .insert<FuncOpLowering, AssertOpLowering, LoadOpLowering, ApplyOpLowering,
              ReturnOpLowering, AccessOpLowering, StoreOpLowering>(
          typeConveter);
}

//===----------------------------------------------------------------------===//
// Stencil Type Converter
//===----------------------------------------------------------------------===//

StencilTypeConverter::StencilTypeConverter(MLIRContext *context_)
    : context(context_) {
  // Add a type conversion for the stencil field type
  addConversion([&](FieldType type) {
    return MemRefType::get(type.getMemRefShape(), type.getElementType());
  });
  addConversion([&](Type type) -> Optional<Type> {
    if (auto fieldType = type.dyn_cast<FieldType>())
      return llvm::None;
    return type;
  });
}

//===----------------------------------------------------------------------===//
// Stencil Pattern Base Class
//===----------------------------------------------------------------------===//

StencilToStdPattern::StencilToStdPattern(StringRef rootOpName,
                                         StencilTypeConverter &typeConverter_,
                                         PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter_.getContext()),
      typeConverter(typeConverter_) {}

Index StencilToStdPattern::computeStrides(ArrayRef<int64_t> shape) const {
  Index result(shape.size());
  result[0] = 1;
  for (size_t i = 1, e = result.size(); i != e; ++i)
    result[i] = result[i - 1] * shape[i - 1];
  return result;
}

Optional<MemRefType> StencilToStdPattern::computeResultType(
    MemRefType inputType, SmallVector<bool, 3> hasAlloc, ShapeOp assertOp,
    ShapeOp accessOp) const {
  if (accessOp.hasShape()) {
    // Compute strides and shape
    auto strides = computeStrides(inputType.getShape());
    auto shape = computeOpShape(accessOp.getOperation(), hasAlloc).getValue();
    assert(strides.size() == shape.size() &&
           "expected strides and shape to have the same size");
    assert(shape.size() <= hasAlloc.size() &&
           "expected shape to have at most allocation rank");

    // Compute the stride considering only the allocated dimensions
    int64_t offset = 0;
    for (auto en : llvm::enumerate(hasAlloc)) {
      if (en.value())
        offset +=
            (accessOp.getLB()[en.index()] - assertOp.getLB()[en.index()]) *
            strides[en.index()];
    }

    // Compute the affine map and assemble the memref type
    auto affineMap =
        makeStridedLinearLayoutMap(strides, offset, accessOp.getContext());
    return MemRefType::get(shape, inputType.getElementType(), affineMap, 0);
  }
  return llvm::None;
}

Optional<Index>
StencilToStdPattern::computeOpShape(Operation *operation,
                                    SmallVector<bool, 3> hasAlloc) const {
  if (auto shapeOp = dyn_cast<ShapeOp>(operation)) {
    if (shapeOp.hasShape()) {
      auto shape = applyFunElementWise(shapeOp.getUB(), shapeOp.getLB(),
                                       std::minus<int64_t>());
      // Keep only the allocated dimensions
      Index allocShape;
      for (auto en : llvm::enumerate(shape))
        if (hasAlloc[en.index()])
          allocShape.push_back(en.value());
      return allocShape;
    }
  }
  return llvm::None;
}

} // namespace stencil
} // namespace mlir

std::unique_ptr<Pass> mlir::createConvertStencilToStandardPass() {
  return std::make_unique<StencilToStandardPass>();
}
