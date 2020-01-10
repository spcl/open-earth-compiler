#include "Conversion/StencilToStandard/ConvertStencilToStandard.h"
#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
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
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>

using namespace mlir;

namespace {

// Helper method getting the parent loop nest
SmallVector<AffineForOp, 3> getLoopNest(Operation *operation) {
  SmallVector<AffineForOp, 3> result;
  Operation *current = operation;
  while (AffineForOp loop = current->getParentOfType<AffineForOp>()) {
    current = loop.getOperation();
    result.push_back(loop);
  }
  return result;
}

// Helper method to check if lower bound is zero
bool isZero(ArrayRef<int64_t> offset) {
  return llvm::any_of(offset, [](int64_t x) { return x != 0; });
}

// Helper method computing the strides given the size
SmallVector<int64_t, 3> computeStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 3> result(shape.size());
  result[0] = 1;
  for (size_t i = 1, e = result.size(); i != e; ++i)
    result[i] = result[i - 1] * shape[i - 1];
  return result;
}

// Helper method computing the shape given the range
SmallVector<int64_t, 3> computeShape(ArrayRef<int64_t> begin,
                                     ArrayRef<int64_t> end) {
  assert(begin.size() == end.size() && "expected bounds to have the same size");
  SmallVector<int64_t, 3> result(begin.size());
  llvm::transform(llvm::zip(end, begin), result.begin(),
                  [](std::tuple<int64_t, int64_t> x) {
                    return std::get<0>(x) - std::get<1>(x);
                  });
  return result;
}

// Helper method computing linearizing the offset
int64_t computeOffset(ArrayRef<int64_t> offset, ArrayRef<int64_t> strides) {
  assert(offset.size() == strides.size() &&
         "expected offset and strides to have the same size");
  int64_t result = 0;
  for (size_t i = 0, e = strides.size(); i != e; ++i) {
    result += offset[i] * strides[i];
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Rewriting Pattern
//===----------------------------------------------------------------------===//

class FuncOpLowering : public ConversionPattern {
public:
  explicit FuncOpLowering(MLIRContext *context)
      : ConversionPattern(FuncOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto funcOp = cast<FuncOp>(operation);

    // Convert the input types
    SmallVector<Type, 10> inputTypes;
    for (auto argument : funcOp.getArguments()) {
      Type argType = argument->getType();
      // Verify no view types
      if (argType.getKind() == stencil::StencilTypes::View) {
        operation->emitError("unexpected argument type '") << argType << "'";
        return matchFailure();
      }

      // Compute the input types of the converted stencil program
      if (argType.getKind() == stencil::StencilTypes::Field) {
        Type inputType = NoneType();
        for (auto &use : argument->getUses()) {
          if (auto assertOp = dyn_cast<stencil::AssertOp>(use.getOwner())) {
            if (isZero(assertOp.getLB())) {
              operation->emitError("expected zero lower bound");
              return matchFailure();
            }
            Type elementType =
                argType.cast<stencil::FieldType>().getElementType();
            ArrayRef<int64_t> shape = assertOp.getUB();
            ArrayRef<int64_t> strides = computeStrides(shape);
            inputType = MemRefType::get(
                shape, elementType,
                makeStridedLinearLayoutMap(strides, 0, rewriter.getContext()),
                0);
            break;
          }
        }
        if (inputType == NoneType()) {
          operation->emitError("failed to find stencil assert for input field");
          return matchFailure();
        }
        inputTypes.push_back(inputType);
      } else {
        inputTypes.push_back(argType);
      }
    }
    if (funcOp.getNumResults() > 0) {
      operation->emitError("expected stencil programs return void");
      return matchFailure();
    }

    // Compute replacement function
    auto replacementType = rewriter.getFunctionType(inputTypes, {});
    auto replacementSymbol =
        funcOp.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    auto replacementOp = rewriter.create<FuncOp>(
        loc, replacementSymbol.getValue(), replacementType, llvm::None);

    // Replace the function body
    Block *entryBlock = replacementOp.addEntryBlock();
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i)
      funcOp.getArgument(i)->replaceAllUsesWith(entryBlock->getArgument(i));
    auto &operations =
        funcOp.getOperation()->getRegion(0).front().getOperations();
    entryBlock->getOperations().splice(entryBlock->begin(), operations);

    // Erase the original function op
    rewriter.eraseOp(operation);
    return matchSuccess();
  }
};

class AssertOpLowering : public ConversionPattern {
public:
  explicit AssertOpLowering(MLIRContext *context)
      : ConversionPattern(stencil::AssertOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(operation);
    return matchSuccess();
  }
};

class LoadOpLowering : public ConversionPattern {
public:
  explicit LoadOpLowering(MLIRContext *context)
      : ConversionPattern(stencil::LoadOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the load operation with a memref cast
    auto loc = operation->getLoc();
    auto loadOp = cast<stencil::LoadOp>(operation);

    // Compute the replacement type
    auto inputType = loadOp.field().getType().cast<MemRefType>();
    Type elementType = inputType.getElementType();
    ArrayRef<int64_t> shape = computeShape(loadOp.getLB(), loadOp.getUB());
    ArrayRef<int64_t> strides = computeStrides(inputType.getShape());
    int64_t offset = computeOffset(loadOp.getLB(), strides);
    auto outputType = MemRefType::get(
        shape, elementType,
        makeStridedLinearLayoutMap(strides, offset, rewriter.getContext()), 0);

    // Replace the load op
    auto subViewOp = rewriter.create<SubViewOp>(loc, outputType, operands[0]);
    operation->getResult(0).replaceAllUsesWith(subViewOp.getResult());
    rewriter.eraseOp(operation);
    return matchSuccess();
  }
};

class ReturnOpLowering : public ConversionPattern {
public:
  explicit ReturnOpLowering(MLIRContext *context)
      : ConversionPattern(stencil::ReturnOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto returnOp = cast<stencil::ReturnOp>(operation);

    // Get the affine loops
    SmallVector<AffineForOp, 3> loops = getLoopNest(operation);
    SmallVector<Value, 3> loopIVs(loops.size(), nullptr);
    llvm::transform(loops, loopIVs.begin(), [](AffineForOp affineForOp) {
      return affineForOp.getInductionVar();
    });
    if (loops.empty())
      return matchFailure();

    // Get temporary buffers
    SmallVector<Operation *, 10> allocOps;
    Operation *currentOp = loops.back().getOperation();
    for (unsigned i = 0, e = returnOp.getNumOperands(); i != e; ++i) {
      currentOp = currentOp->getPrevNode();
      allocOps.push_back(currentOp);
      assert(dyn_cast<AllocOp>(currentOp) &&
             "failed to find allocation for results");
    }
    SmallVector<Value, 10> allocVals(allocOps.size());
    llvm::transform(llvm::reverse(allocOps), allocVals.begin(),
                    [](Operation *allocOp) { return allocOp->getResult(0); });

    // Replace the return op by store ops
    for (unsigned i = 0, e = returnOp.getNumOperands(); i != e; ++i) {
      rewriter.create<AffineStoreOp>(loc, returnOp.getOperand(i), allocVals[i],
                                     loopIVs);
    }
    rewriter.eraseOp(operation);
    return matchSuccess();
  }
};

class ApplyOpLowering : public ConversionPattern {
public:
  explicit ApplyOpLowering(MLIRContext *context)
      : ConversionPattern(stencil::ApplyOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto applyOp = cast<stencil::ApplyOp>(operation);

    // Verify the the lower bound is zero
    if (isZero(applyOp.getLB())) {
      operation->emitError("expected zero lower bound");
      return matchFailure();
    }

    // Allocate and deallocate storage for every output
    for (unsigned i = 0, e = applyOp.getNumResults(); i != e; ++i) {
      Type elementType = applyOp.getResultViewType(i).getElementType();
      ArrayRef<int64_t> shape = applyOp.getUB();
      ArrayRef<int64_t> strides = computeStrides(shape);
      auto allocType = MemRefType::get(
          shape, elementType,
          makeStridedLinearLayoutMap(strides, 0, rewriter.getContext()), 0);
      auto allocOp = rewriter.create<AllocOp>(loc, allocType);
      applyOp.getResult(i)->replaceAllUsesWith(allocOp.getResult());
      auto returnOp = allocOp.getParentRegion()->back().getTerminator();
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<DeallocOp>(loc, allocOp.getResult());
      rewriter.setInsertionPointAfter(allocOp);
    }

    // Generate the apply loop nest
    ArrayRef<int64_t> upper = applyOp.getUB();
    assert(upper.size() >= 1 && "expected bounds to at least one dimension");
    AffineForOp loop;
    for (size_t i = 0, e = upper.size(); i != e; ++i) {
      loop = rewriter.create<AffineForOp>(loc, 0, upper.rbegin()[i]);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Forward the apply operands and copy the body
    BlockAndValueMapping mapper;
    for (size_t i = 0, e = applyOp.operands().size(); i < e; ++i) {
      mapper.map(applyOp.getBody()->getArgument(i), applyOp.getOperand(i));
    }
    for (auto &op : applyOp.getBody()->getOperations()) {
      rewriter.clone(op, mapper);
    }
    rewriter.eraseOp(applyOp);
    return matchSuccess();
  }
}; // namespace

class AccessOpLowering : public ConversionPattern {
public:
  explicit AccessOpLowering(MLIRContext *context)
      : ConversionPattern(stencil::AccessOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto accessOp = cast<stencil::AccessOp>(operation);

    // Get the affine loops
    SmallVector<AffineForOp, 3> loops = getLoopNest(operation);
    SmallVector<Value, 3> loopIVs(loops.size());
    llvm::transform(loops, loopIVs.begin(), [](AffineForOp affineForOp) {
      return affineForOp.getInductionVar();
    });
    if (loops.empty()) {
      return matchFailure();
    }
    assert(loops.size() == accessOp.getOffset().size() &&
           "expected loop nest and access offset to have the same size");

    // Compute the access offsets
    auto addExpr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
    auto addMap = AffineMap::get(2, 0, addExpr);
    auto accessOffset = accessOp.getOffset();
    SmallVector<Value, 3> loadOffset;
    for (size_t i = 0, e = accessOffset.size(); i != e; ++i) {
      auto constantOp = rewriter.create<ConstantIndexOp>(loc, accessOffset[i]);
      ValueRange addParams = {loopIVs[i], constantOp.getResult()};
      auto affineApplyOp =
          rewriter.create<AffineApplyOp>(loc, addMap, addParams);
      loadOffset.push_back(affineApplyOp.getResult());
    }

    // Replace the access op by a load op
    rewriter.replaceOpWithNewOp<AffineLoadOp>(operation, accessOp.view(),
                                              loadOffset);                                
    return matchSuccess();
  }
};

class StoreOpLowering : public ConversionPattern {
public:
  explicit StoreOpLowering(MLIRContext *context)
      : ConversionPattern(stencil::StoreOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto storeOp = cast<stencil::StoreOp>(operation);

    // Compute the replacement type
    auto inputType = storeOp.field().getType().cast<MemRefType>();
    Type elementType = inputType.getElementType();
    ArrayRef<int64_t> shape = computeShape(storeOp.getLB(), storeOp.getUB());
    ArrayRef<int64_t> strides = computeStrides(inputType.getShape());
    int64_t offset = computeOffset(storeOp.getLB(), strides);
    auto outputType = MemRefType::get(
        shape, elementType,
        makeStridedLinearLayoutMap(strides, offset, rewriter.getContext()), 0);

    // Remove allocation and deallocation and insert memref cast
    auto allocOp = storeOp.view()->getDefiningOp();
    rewriter.setInsertionPoint(allocOp);
    auto subViewOp =
        rewriter.create<SubViewOp>(loc, outputType, storeOp.field());
    allocOp->getResult(0).replaceAllUsesWith(subViewOp.getResult());
    rewriter.eraseOp(allocOp);
    for (auto &use : storeOp.view()->getUses()) {
      if (auto deallocOp = dyn_cast<DeallocOp>(use.getOwner()))
        rewriter.eraseOp(deallocOp);
    }
    rewriter.eraseOp(operation);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

class StencilToStandardTarget : public ConversionTarget {
public:
  explicit StencilToStandardTarget(MLIRContext &context)
      : ConversionTarget(context) {}

  bool isDynamicallyLegal(Operation *op) const override {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      return !funcOp.getAttr(
                 stencil::StencilDialect::getStencilProgramAttrName()) &&
             !funcOp.getAttr(
                 stencil::StencilDialect::getStencilFunctionAttrName());
    } else
      return true;
  }
};

//===----------------------------------------------------------------------===//
// Rewriting Pass
//===----------------------------------------------------------------------===//

struct StencilToStandardPass : public ModulePass<StencilToStandardPass> {
  void runOnModule() override;
};

void StencilToStandardPass::runOnModule() {
  OwningRewritePatternList patterns;
  auto module = getModule();

  populateStencilToStandardConversionPatterns(patterns, module.getContext());

  StencilToStandardTarget target(*(module.getContext()));
  target.addLegalDialect<AffineOpsDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addDynamicallyLegalOp<FuncOp>();

  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

} // namespace

void mlir::populateStencilToStandardConversionPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *ctx) {
  patterns
      .insert<FuncOpLowering, AssertOpLowering, LoadOpLowering, ApplyOpLowering,
              AccessOpLowering, StoreOpLowering, ReturnOpLowering>(ctx);
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::stencil::createConvertStencilToStandardPass() {
  return std::make_unique<StencilToStandardPass>();
}

static PassRegistration<StencilToStandardPass>
    pass("convert-stencil-to-standard",
         "Convert stencil dialect to standard operations");
