#include "Conversion/StencilToStandard/ConvertStencilToStandard.h"
#include "Conversion/StencilToStandard/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
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
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>

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
  return llvm::all_of(offset, [](int64_t x) {
    return x == 0 || x == stencil::kIgnoreDimension;
  });
}

// Helper to filter ignored dimensions
SmallVector<int64_t, 3> filterIgnoredDimensions(ArrayRef<int64_t> offset) {
  SmallVector<int64_t, 3> filtered(llvm::make_range(offset.begin(), offset.end()));
  llvm::erase_if(filtered, [](int64_t x) {
    return x == stencil::kIgnoreDimension;
  });
  return filtered;
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
  // Remove ignored dimensions
  llvm::erase_if(result, [](int64_t x) { return x == 0; });
  return result;
}

// Helper method computing linearizing the offset
int64_t computeOffset(ArrayRef<int64_t> offset, ArrayRef<int64_t> strides) {
  SmallVector<int64_t, 3> filtered = filterIgnoredDimensions(offset);
  // Delete the ignored dimensions
  assert(filtered.size() == strides.size() &&
         "expected offset and strides to have the same size");

  // Compute the linear offset
  int64_t result = 0;
  for (size_t i = 0, e = strides.size(); i != e; ++i) {
    result += filtered[i] * strides[i];
  }
  return result;
}

// Helper to compute a memref type
MemRefType computeMemRefType(Type elementType, ArrayRef<int64_t> shape,
                             ArrayRef<int64_t> strides,
                             ArrayRef<int64_t> origin,
                             ConversionPatternRewriter &rewriter) {
  // Get the element type
  int64_t offset = 0;
  if (origin.size() != 0) {
    offset = computeOffset(origin, strides);
  }
  auto map = makeStridedLinearLayoutMap(strides, offset, rewriter.getContext());
  return MemRefType::get(shape, elementType, map, 0);
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
      Type argType = argument.getType();
      // Verify no view types
      if (argType.isa<stencil::ViewType>()) {
        funcOp.emitOpError("unexpected argument type '") << argType << "'";
        return matchFailure();
      }

      // Compute the input types of the converted stencil program
      if (argType.isa<stencil::FieldType>()) {
        Type inputType = NoneType();
        for (auto &use : argument.getUses()) {
          if (auto assertOp = dyn_cast<stencil::AssertOp>(use.getOwner())) {
            auto shape = filterIgnoredDimensions(assertOp.getUB());
            inputType =
                computeMemRefType(assertOp.getFieldType().getElementType(),
                                  shape, computeStrides(shape), {}, rewriter);
            break;
          }
        }
        if (inputType == NoneType()) {
          funcOp.emitOpError("failed to convert argument types");
          return matchFailure();
        }
        inputTypes.push_back(inputType);
      } else {
        inputTypes.push_back(argType);
      }
    }
    if (funcOp.getNumResults() > 0) {
      operation->emitOpError("expected program to return void");
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
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      funcOp.getArgument(i).replaceAllUsesWith(entryBlock->getArgument(i));
    }
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
    auto assertOp = cast<stencil::AssertOp>(operation);

    // Verify the field has been converted
    if (!assertOp.field().getType().isa<MemRefType>())
      return matchFailure();

    // Verify the assert op has lower bound zero
    if (!isZero(assertOp.getLB())) {
      assertOp.emitOpError("expected zero lower bound");
      return matchFailure();
    }

    // Check if the field is large enough
    auto verifyBounds = [&](const SmallVector<int64_t, 3> &lb,
                            const SmallVector<int64_t, 3> &ub) {
      if (llvm::any_of(llvm::zip(lb, assertOp.getLB()),
                       [](std::tuple<int64_t, int64_t> x) {
                         return std::get<0>(x) < std::get<1>(x) &&
                                std::get<0>(x) != stencil::kIgnoreDimension;
                       }) ||
          llvm::any_of(llvm::zip(ub, assertOp.getUB()),
                       [](std::tuple<int64_t, int64_t> x) {
                         return std::get<0>(x) > std::get<1>(x) &&
                                std::get<1>(x) != stencil::kIgnoreDimension;
                       }))
        return false;
      return true;
    };
    for (OpOperand &use : assertOp.field().getUses()) {
      if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner())) {
        if (llvm::is_contained(storeOp.getLB(), stencil::kIgnoreDimension)) {
          storeOp.emitOpError("field expected to have full dimensionality");
          return matchFailure();
        }
        if (!verifyBounds(storeOp.getLB(), storeOp.getUB())) {
          storeOp.emitOpError("field bounds not large enough");
          return matchFailure();
        }
      }
      if (auto loadOp = dyn_cast<stencil::LoadOp>(use.getOwner())) {
        if (loadOp.lb().hasValue() && loadOp.ub().hasValue())
          if (!verifyBounds(loadOp.getLB(), loadOp.getUB())) {
            loadOp.emitOpError("field bounds not large enough");
            return matchFailure();
          }
      }
    }

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
    // Replace the load operation with a subview op
    auto loc = operation->getLoc();
    auto loadOp = cast<stencil::LoadOp>(operation);

    // Verify the field has been converted
    if (!loadOp.field().getType().isa<MemRefType>())
      return matchFailure();

    // Compute the replacement types
    auto inputType = loadOp.field().getType().cast<MemRefType>();
    auto shape = computeShape(loadOp.getLB(), loadOp.getUB());
    auto strides = computeStrides(inputType.getShape());
    assert(shape.size() == inputType.getRank() &&
           strides.size() == inputType.getRank() &&
           "expected input field shape and strides to have the same rank");
    auto outputType =
        computeMemRefType(inputType.getElementType(), shape, strides,
                          filterIgnoredDimensions(loadOp.getLB()), rewriter);

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

    // Verify the arguments have been converted
    if (llvm::any_of(
            llvm::zip(applyOp.getBody()->getArguments(), applyOp.getOperands()),
            [](std::tuple<Value, Value> x) {
              return std::get<0>(x).getType().isa<stencil::ViewType>() &&
                     !std::get<1>(x).getType().isa<MemRefType>();
            })) {
      return matchFailure();
    }

    // Verify the the lower bound is zero
    if (!isZero(applyOp.getLB())) {
      applyOp.emitOpError("expected zero lower bound");
      return matchFailure();
    }

    // Allocate and deallocate storage for every output
    for (unsigned i = 0, e = applyOp.getNumResults(); i != e; ++i) {
      Type elementType = applyOp.getResultViewType(i).getElementType();
      auto strides = computeStrides(applyOp.getUB());
      auto allocType = computeMemRefType(elementType, applyOp.getUB(), strides,
                                         {}, rewriter);

      auto allocOp = rewriter.create<AllocOp>(loc, allocType);
      applyOp.getResult(i).replaceAllUsesWith(allocOp.getResult());
      auto returnOp = allocOp.getParentRegion()->back().getTerminator();
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<DeallocOp>(loc, allocOp.getResult());
      rewriter.setInsertionPointAfter(allocOp);
    }

    // Generate the apply loop nest
    auto upper = applyOp.getUB();
    assert(upper.size() >= 1 && "expected bounds to at least one dimension");
    for (size_t i = 0, e = upper.size(); i != e; ++i) {
      auto loop = rewriter.create<AffineForOp>(loc, 0, upper.rbegin()[i]);
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

    // Check the view is a memref type
    if (!accessOp.view().getType().isa<MemRefType>())
      return matchFailure();

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
    auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
    auto map = AffineMap::get(2, 0, expr);
    auto offset = accessOp.getOffset();
    SmallVector<Value, 3> loadOffset;
    for (size_t i = 0, e = offset.size(); i != e; ++i) {
      if (offset[i] != stencil::kIgnoreDimension) {
        auto constantOp = rewriter.create<ConstantIndexOp>(loc, offset[i]);
        ValueRange params = {loopIVs[i], constantOp.getResult()};
        auto affineApplyOp = rewriter.create<AffineApplyOp>(loc, map, params);
        loadOffset.push_back(affineApplyOp.getResult());
      }
    }
    assert(loadOffset.size() ==
               accessOp.view().getType().cast<MemRefType>().getRank() &&
           "expected load offset size to match memref rank");

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

    // Verify the field has been converted
    if (!(storeOp.field().getType().isa<MemRefType>() &&
          storeOp.view().getType().isa<MemRefType>()))
      return matchFailure();

    // Compute the replacement types
    auto inputType = storeOp.field().getType().cast<MemRefType>();
    assert(storeOp.getLB().size() == inputType.getRank() &&
           "expected lower bounds and memref to have the same rank");
    auto shape = computeShape(storeOp.getLB(), storeOp.getUB());
    auto strides = computeStrides(inputType.getShape());
    auto outputType = computeMemRefType(inputType.getElementType(), shape,
                                        strides, storeOp.getLB(), rewriter);

    // Remove allocation and deallocation and insert subview op
    auto allocOp = storeOp.view().getDefiningOp();
    rewriter.setInsertionPoint(allocOp);
    auto subViewOp =
        rewriter.create<SubViewOp>(loc, outputType, storeOp.field());
    allocOp->getResult(0).replaceAllUsesWith(subViewOp.getResult());
    rewriter.eraseOp(allocOp);
    for (auto &use : storeOp.view().getUses()) {
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
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  if (failed(applyFullConversion(module, target, patterns))) {
    signalPassFailure();
  }
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
