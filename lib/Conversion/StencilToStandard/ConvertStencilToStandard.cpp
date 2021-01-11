#include "Conversion/StencilToStandard/ConvertStencilToStandard.h"
#include "Conversion/StencilToStandard/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
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
        FunctionType::get(funcOp.getContext(), result.getConvertedTypes(),
                          funcOp.getType().getResults());

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

class YieldOpLowering : public StencilOpToStdPattern<scf::YieldOp> {
public:
  using StencilOpToStdPattern<scf::YieldOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto yieldOp = cast<scf::YieldOp>(operation);

    // Remove all result types from the operand list
    SmallVector<Value, 4> newOperands;
    llvm::copy_if(
        yieldOp.getOperands(), std::back_inserter(newOperands),
        [](Value value) { return !value.getType().isa<ResultType>(); });
    assert(newOperands.size() < yieldOp.getNumOperands() &&
           "expected if op to return results");

    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newOperands);
    return success();
  }
};

class IfOpLowering : public StencilOpToStdPattern<scf::IfOp> {
public:
  using StencilOpToStdPattern<scf::IfOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto ifOp = cast<scf::IfOp>(operation);

    // Remove all result types from the result list
    SmallVector<Type, 4> newTypes;
    llvm::copy_if(ifOp.getResultTypes(), std::back_inserter(newTypes),
                  [](Type type) { return !type.isa<ResultType>(); });
    assert(newTypes.size() < ifOp.getNumResults() &&
           "expected if op to return results");

    // Create a new if op and move the bodies
    auto newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newTypes,
                                            ifOp.condition(), true);
    newOp.walk([&](scf::YieldOp yieldOp) { rewriter.eraseOp(yieldOp); });
    rewriter.mergeBlocks(ifOp.getBody(0), newOp.getBody(0), llvm::None);
    rewriter.mergeBlocks(ifOp.getBody(1), newOp.getBody(1), llvm::None);

    // Erase the if op if there are no results to replace
    if (newOp.getNumResults() == 0) {
      rewriter.eraseOp(ifOp);
      return success();
    }

    // Replace the if op by the results of the new op
    SmallVector<Value, 4> newResults(ifOp.getNumResults(),
                                     newOp.getResults().front());
    auto it = newOp.getResults().begin();
    for (auto en : llvm::enumerate(ifOp.getResultTypes())) {
      if (!en.value().isa<ResultType>())
        newResults[en.index()] = *it++;
    }
    rewriter.replaceOp(ifOp, newResults);
    return success();
  }
};

class CastOpLowering : public StencilOpToStdPattern<stencil::CastOp> {
public:
  using StencilOpToStdPattern<stencil::CastOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
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

    // Compute the shape of the subview
    auto subViewShape =
        computeSubViewShape(fieldType, operation, valueToLB[loadOp.field()]);

    // Replace the load op by a subview op
    auto subViewOp = rewriter.create<SubViewOp>(
        loc, operands[0], std::get<0>(subViewShape), std::get<1>(subViewShape),
        std::get<2>(subViewShape), ValueRange(), ValueRange(), ValueRange());
    rewriter.replaceOp(operation, subViewOp.getResult());
    return success();
  }
};

class BufferOpLowering : public StencilOpToStdPattern<stencil::BufferOp> {
public:
  using StencilOpToStdPattern<stencil::BufferOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto bufferOp = cast<stencil::BufferOp>(operation);

    // Free the buffer memory after the last use
    assert(isa<gpu::AllocOp>(operands[0].getDefiningOp()) &&
           "expected the temporary points to an allocation");
    Operation *lastUser = bufferOp.getOperation();
    for (auto user : bufferOp.getResult().getUsers()) {
      if (lastUser->isBeforeInBlock(user))
        lastUser = user;
    }
    rewriter.setInsertionPointAfter(lastUser);
    rewriter.create<gpu::DeallocOp>(loc, TypeRange(),
                                    ValueRange(bufferOp.temp()));

    rewriter.replaceOp(operation, bufferOp.temp());
    return success();
  }
};

class ApplyOpLowering : public StencilOpToStdPattern<stencil::ApplyOp> {
public:
  using StencilOpToStdPattern<stencil::ApplyOp>::StencilOpToStdPattern;

  // Get the temporary and the shape of the buffer
  std::tuple<Value, ShapeOp> getShapeAndTemporary(Value value) const {
    if (auto storeOp = getUserOp<stencil::StoreOp>(value)) {
      return std::make_tuple(storeOp.temp(),
                             cast<ShapeOp>(storeOp.getOperation()));
    }
    if (auto bufferOp = getUserOp<stencil::BufferOp>(value)) {
      return std::make_tuple(bufferOp.temp(),
                             cast<ShapeOp>(bufferOp.getOperation()));
    }
    llvm_unreachable("expected a valid storage operation");
    return std::make_tuple(nullptr, nullptr);
  }

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto applyOp = cast<stencil::ApplyOp>(operation);
    auto shapeOp = cast<ShapeOp>(operation);

    // Allocate storage for buffers or introduce get a view of the output field
    SmallVector<Value, 10> newResults;
    for (auto result : applyOp.getResults()) {
      Value temp;
      ShapeOp shapeOp;
      std::tie(temp, shapeOp) = getShapeAndTemporary(result);
      auto oldType = temp.getType().cast<TempType>();
      auto tempType =
          TempType::get(oldType.getElementType(), oldType.getAllocation(),
                        shapeOp.getLB(), shapeOp.getUB());
      auto allocType = typeConverter.convertType(tempType).cast<MemRefType>();
      assert(allocType.hasStaticShape() &&
             "expected buffer to have a static shape");
      auto segAttr = rewriter.getNamedAttr(
          "operand_segment_sizes", rewriter.getI32VectorAttr({0, 0, 0}));
      auto allocOp = rewriter.create<gpu::AllocOp>(loc, TypeRange(allocType),
                                                   ValueRange(), segAttr);
      newResults.push_back(allocOp.getResult(0));
    }

    // Compute the loop bounds starting from zero
    // (in case of loop unrolling adjust the step of the loop)
    SmallVector<Value, 3> lbs, ubs, steps;
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    for (int64_t i = 0, e = shapeOp.getRank(); i != e; ++i) {
      int64_t lb = shapeOp.getLB()[i];
      int64_t ub = shapeOp.getUB()[i];
      int64_t step = returnOp.unroll().hasValue() ? returnOp.getUnroll()[i] : 1;
      lbs.push_back(rewriter.create<ConstantIndexOp>(loc, lb));
      ubs.push_back(rewriter.create<ConstantIndexOp>(loc, ub));
      steps.push_back(rewriter.create<ConstantIndexOp>(loc, step));
    }

    // Convert the signature of the apply op body
    // (access the apply op operands and introduce the loop indicies)
    TypeConverter::SignatureConversion result(applyOp.getNumOperands());
    for (auto &en : llvm::enumerate(applyOp.getOperands())) {
      result.remapInput(en.index(), operands[en.index()]);
    }
    rewriter.applySignatureConversion(&applyOp.region(), result);

    // Affine map used for induction variable computation
    // TODO this is only useful for sequential loops
    auto fwdExpr = rewriter.getAffineDimExpr(0);
    auto fwdMap = AffineMap::get(1, 0, fwdExpr);

    // Replace the stencil apply operation by a loop nest
    auto parallelOp = rewriter.create<ParallelOp>(loc, lbs, ubs, steps);
    rewriter.mergeBlockBefore(
        applyOp.getBody(),
        parallelOp.getLoopBody().getBlocks().back().getTerminator());

    // Insert index variables at the beginning of the loop body
    rewriter.setInsertionPointToStart(parallelOp.getBody());
    for (int64_t i = 0, e = shapeOp.getRank(); i != e; ++i) {
      rewriter.create<AffineApplyOp>(
          loc, fwdMap, ValueRange(parallelOp.getInductionVars()[i]));
    }

    // Replace the applyOp
    rewriter.replaceOp(applyOp, newResults);
    return success();
  }
}; // namespace

class StoreResultOpLowering
    : public StencilOpToStdPattern<stencil::StoreResultOp> {
public:
  using StencilOpToStdPattern<stencil::StoreResultOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto resultOp = cast<stencil::StoreResultOp>(operation);

    // Iterate over all return op operands of the result
    for (auto opOperand : valueToReturnOpOperands[resultOp.res()]) {
      // Get the return op and the parallel op
      auto returnOp = cast<stencil::ReturnOp>(opOperand->getOwner());
      auto parallelOp = returnOp->getParentOfType<ParallelOp>();

      // Check the parent has been lowered
      if (!isa<ParallelOp>(returnOp->getParentOp()))
        return failure();

      // Store the result in case there is something to store
      if (resultOp.operands().size() == 1) {
        // Compute unroll factor and dimension
        auto unrollFac = returnOp.getUnrollFac();
        size_t unrollDim = returnOp.getUnrollDim();

        // Get the output buffer
        gpu::AllocOp allocOp;
        unsigned bufferCount = (returnOp.getNumOperands() / unrollFac) -
                               (opOperand->getOperandNumber() / unrollFac);
        auto *node = parallelOp.getOperation();
        while (bufferCount != 0 && (node = node->getPrevNode())) {
          if ((allocOp = dyn_cast<gpu::AllocOp>(node)))
            bufferCount--;
        }
        assert(bufferCount == 0 && "expected valid buffer allocation");

        // Compute the static store offset
        auto lb = valueToLB[opOperand->get()];
        llvm::transform(lb, lb.begin(), std::negate<int64_t>());
        lb[unrollDim] += opOperand->getOperandNumber() % unrollFac;

        // Set the insertion point to the defining op if possible
        auto result = resultOp.operands().front();
        if (result.getDefiningOp() &&
            result.getDefiningOp()->getParentOp() == resultOp->getParentOp())
          rewriter.setInsertionPointAfter(result.getDefiningOp());

        // Compute the index values and introduce the store operation
        auto inductionVars = getInductionVars(operation);
        SmallVector<bool, 3> allocation(lb.size(), true);
        auto storeOffset =
            computeIndexValues(inductionVars, lb, allocation, rewriter);
        rewriter.create<mlir::StoreOp>(loc, result, allocOp.getResult(0),
                                       storeOffset);
      }
    }

    rewriter.eraseOp(operation);
    return success();
  }
};

class ReturnOpLowering : public StencilOpToStdPattern<stencil::ReturnOp> {
public:
  using StencilOpToStdPattern<stencil::ReturnOp>::StencilOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
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

    // Compute the shape of the subview
    auto subViewShape =
        computeSubViewShape(fieldType, operation, valueToLB[storeOp.field()]);

    // Replace the allocation by a subview
    auto allocOp = operands[0].getDefiningOp();
    rewriter.setInsertionPoint(allocOp);
    auto subViewOp = rewriter.create<SubViewOp>(
        loc, operands[1], std::get<0>(subViewShape), std::get<1>(subViewShape),
        std::get<2>(subViewShape), ValueRange(), ValueRange(), ValueRange());
    rewriter.replaceOp(allocOp, subViewOp.getResult());
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
      return !StencilDialect::isStencilProgram(funcOp);
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return llvm::none_of(ifOp.getResultTypes(),
                           [](Type type) { return type.isa<ResultType>(); });
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      return llvm::none_of(yieldOp.getOperandTypes(),
                           [](Type type) { return type.isa<ResultType>(); });
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

private:
  Index findLB(Value value);
};

Index StencilToStandardPass::findLB(Value value) {
  SmallVector<Operation *> operations(value.getUsers().begin(),
                                      value.getUsers().end());
  if (auto definingOp = value.getDefiningOp())
    operations.push_back(definingOp);
  // Search the lower bound of the value
  for (auto op : operations) {
    if (auto loadOp = dyn_cast<stencil::LoadOp>(op))
      return cast<ShapeOp>(loadOp.getOperation()).getLB();
    if (auto storeOp = dyn_cast<stencil::StoreOp>(op))
      return cast<ShapeOp>(storeOp.getOperation()).getLB();
    if (auto bufferOp = dyn_cast<stencil::BufferOp>(op))
      return cast<ShapeOp>(bufferOp.getOperation()).getLB();
  }
  return {};
}

void StencilToStandardPass::runOnOperation() {
  OwningRewritePatternList patterns;
  auto module = getOperation();

  // Check all shapes are set
  auto shapeResult = module.walk([&](ShapeOp shapeOp) {
    if (!shapeOp.hasShape()) {
      shapeOp.emitOpError("expected to have a valid shape");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (shapeResult.wasInterrupted()) {
    return signalPassFailure();
  }

  // Store the input bounds of the stencil program
  DenseMap<Value, Index> valueToLB;
  module.walk([&](stencil::CastOp castOp) {
    valueToLB[castOp.res()] = cast<ShapeOp>(castOp.getOperation()).getLB();
  });
  module.walk([&](stencil::ApplyOp applyOp) {
    auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
    // Store the lower bounds for all arguments
    for (auto en : llvm::enumerate(applyOp.getOperands())) {
      valueToLB[applyOp.getBody()->getArgument(en.index())] =
          findLB(en.value());
    }
    // Store the lower bounds for all results
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    for (auto en : llvm::enumerate(applyOp.getResults())) {
      Index lb = findLB(en.value());
      assert(lb.size() == shapeOp.getRank() &&
             "expected to find valid storage shape");
      // Store the bound for all return op operands writting to the result
      unsigned unrollFac = returnOp.getUnrollFac();
      for (unsigned i = 0, e = unrollFac; i != e; ++i) {
        valueToLB[returnOp.getOperand(en.index() * unrollFac + i)] = lb;
      }
    }
  });

  // Check there is exactly one storage operation per apply op result
  auto uniqueStorageResult = module.walk([&](stencil::ApplyOp applyOp) {
    for (auto result : applyOp.getResults()) {
      unsigned storageOps = 0;
      for (auto user : result.getUsers()) {
        if (isa<stencil::BufferOp>(user) || isa<stencil::StoreOp>(user)) {
          storageOps++;
        }
      }
      if (storageOps != 1) {
        applyOp.emitOpError("expected apply op results to have storage");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (uniqueStorageResult.wasInterrupted())
    return signalPassFailure();

  // Store the return op operands for the result values
  DenseMap<Value, SmallVector<OpOperand *, 10>> valueToReturnOpOperands;
  auto storeMappingResult = module.walk([&](stencil::StoreResultOp resultOp) {
    if (!resultOp.getReturnOpOperands()) {
      resultOp.emitOpError("expected valid return op operands");
      return WalkResult::interrupt();
    }
    valueToReturnOpOperands[resultOp.res()] =
        resultOp.getReturnOpOperands().getValue();
    return WalkResult::advance();
  });
  if (storeMappingResult.wasInterrupted())
    return signalPassFailure();

  StencilTypeConverter typeConverter(module.getContext());
  populateStencilToStdConversionPatterns(typeConverter, valueToLB,
                                         valueToReturnOpOperands, patterns);

  StencilToStdTarget target(*(module.getContext()));
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<SCFDialect>();
  target.addDynamicallyLegalOp<FuncOp>();
  target.addDynamicallyLegalOp<scf::IfOp>();
  target.addDynamicallyLegalOp<scf::YieldOp>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<gpu::AllocOp>();
  target.addLegalOp<gpu::DeallocOp>();
  if (failed(applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace stencil {

// Populate the conversion pattern list
void populateStencilToStdConversionPatterns(
    StencilTypeConverter &typeConveter, DenseMap<Value, Index> &valueToLB,
    DenseMap<Value, SmallVector<OpOperand *, 10>> &valueToReturnOpOperands,
    mlir::OwningRewritePatternList &patterns) {
  patterns.insert<FuncOpLowering, IfOpLowering, YieldOpLowering, CastOpLowering,
                  LoadOpLowering, ApplyOpLowering, BufferOpLowering,
                  ReturnOpLowering, StoreResultOpLowering, AccessOpLowering,
                  DynAccessOpLowering, IndexOpLowering, StoreOpLowering>(
      typeConveter, valueToLB, valueToReturnOpOperands);
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

StencilToStdPattern::StencilToStdPattern(
    StringRef rootOpName, StencilTypeConverter &typeConverter,
    DenseMap<Value, Index> &valueToLB,
    DenseMap<Value, SmallVector<OpOperand *, 10>> &valueToReturnOpOperands,
    PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter.getContext()),
      typeConverter(typeConverter), valueToLB(valueToLB),
      valueToReturnOpOperands(valueToReturnOpOperands) {}

Index StencilToStdPattern::computeShape(ShapeOp shapeOp) const {
  return applyFunElementWise(shapeOp.getUB(), shapeOp.getLB(),
                             std::minus<int64_t>());
}

SmallVector<Value, 3>
StencilToStdPattern::getInductionVars(Operation *operation) const {
  SmallVector<Value, 3> inductionVariables;

  // Get the parallel loop
  auto parallelOp = operation->getParentOfType<ParallelOp>();
  // TODO only useful for sequential applies
  auto forOp = operation->getParentOfType<ForOp>();
  if (!parallelOp)
    return inductionVariables;

  // Collect the induction variables
  parallelOp.walk([&](AffineApplyOp applyOp) {
    for (auto operand : applyOp.getOperands()) {
      // TODO only useful for sequential applies
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
StencilToStdPattern::computeSubViewShape(FieldType fieldType, ShapeOp shapeOp,
                                         Index castLB) const {
  auto shape = computeShape(shapeOp);
  Index revShape, revOffset, revStrides;
  for (auto en : llvm::enumerate(fieldType.getAllocation())) {
    // Insert values at the front to convert from column- to row-major
    if (en.value()) {
      revShape.insert(revShape.begin(), shape[en.index()]);
      revStrides.insert(revStrides.begin(), 1);
      revOffset.insert(revOffset.begin(),
                       shapeOp.getLB()[en.index()] - castLB[en.index()]);
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
