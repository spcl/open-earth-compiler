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
    ArrayRef<NamedAttribute> attrs;
    auto subViewOp = rewriter.create<SubViewOp>(loc, resultType.getValue(),
                                                castOp.getResult(), attrs);
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

    // Get the shape
    auto shapeOp = cast<ShapeOp>(operation);
    if (!shapeOp.hasShape())
      return failure();

    // Compute the loop bounds starting from zero
    // (in case of loop unrolling adjust the step of the loop)
    SmallVector<Value, 3> lb;
    SmallVector<Value, 3> ub;
    SmallVector<Value, 3> steps;
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

    // Get the loop operation
    if (!isa<ParallelOp>(operation->getParentOp()))
      return failure();
    auto loop = cast<ParallelOp>(operation->getParentOp());

    // Get allocations of result buffers
    SmallVector<Value, 10> allocVals;
    unsigned allocCount =
        returnOp.getNumOperands() / returnOp.getUnrollFactor();
    for (auto *it = loop.getOperation(); it && allocVals.size() < allocCount;
         it = it->getPrevNode()) {
      if (auto allocOp = dyn_cast<AllocOp>(it))
        allocVals.insert(allocVals.begin(), allocOp.getResult());
    }
    assert(allocVals.size() == allocCount &&
           "expected allocation for every result of the stencil operator");

    // Replace the return op by store ops
    auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
    auto map = AffineMap::get(2, 0, expr);

    for (unsigned i = 0, e = allocCount; i != e; ++i) {
      for (unsigned j = 0, e = returnOp.getUnrollFactor(); j != e; ++j) {
        rewriter.setInsertionPoint(returnOp);
        unsigned operandIdx = i * returnOp.getUnrollFactor() + j;
        auto definingOp = returnOp.getOperand(operandIdx).getDefiningOp();
        if (definingOp && returnOp.getParentOp() == definingOp->getParentOp())
          rewriter.setInsertionPointAfter(definingOp);

        // TODO subtract the loop loop bounds

        // Add the unrolling offset to the loop ivs
        SmallVector<Value, 3> storeOffset = loop.getInductionVars();
        if (j > 0) {
          auto unroll = returnOp.getUnroll();
          assert(llvm::count(unroll, 1) == unroll.size() - 1 &&
                 "expected a single non-zero entry");
          auto it = llvm::find_if(unroll, [&](int64_t x) {
            return x == returnOp.getUnrollFactor();
          });
          auto unrollDim = std::distance(unroll.begin(), it);
          auto constantOp = rewriter.create<ConstantIndexOp>(loc, j);
          ValueRange params = {loop.getInductionVars()[unrollDim],
                               constantOp.getResult()};
          auto affineApplyOp = rewriter.create<AffineApplyOp>(loc, map, params);
          storeOffset[unrollDim] = affineApplyOp.getResult();
        }
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

    // Get the allocation
    auto tempType = accessOp.temp().getType().cast<TempType>();

    // Compute the access offsets
    auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
    auto map = AffineMap::get(2, 0, expr);
    auto offset = accessOp.getOffset();
    SmallVector<Value, 3> loadOffset;
    for (auto en : llvm::enumerate(tempType.getAllocation())) {
      if (en.value()) {
        ValueRange params = {
            loopOp.getInductionVars()[en.index()],
            rewriter.create<ConstantIndexOp>(loc, offset[en.index()])
                .getResult()};
        auto affineApplyOp = rewriter.create<AffineApplyOp>(loc, map, params);
        loadOffset.push_back(affineApplyOp.getResult());
      }
    }

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

    // Replace the allocation by a subview
    auto allocOp = operands[0].getDefiningOp();
    rewriter.setInsertionPoint(allocOp);
    ArrayRef<NamedAttribute> attrs;
    auto subViewOp = rewriter.create<SubViewOp>(loc, resultType.getValue(),
                                                castOp.getResult(), attrs);
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

  StencilTypeConverter typeConverter(module.getContext());
  populateStencilToStdConversionPatterns(typeConverter, patterns);

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
