#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;
using namespace stencil;

//===----------------------------------------------------------------------===//
// stencil.apply
//===----------------------------------------------------------------------===//

static ParseResult parseApplyOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 8> operands;
  SmallVector<OpAsmParser::OperandType, 8> arguments;
  SmallVector<Type, 8> operandTypes;

  // Parse the assignment list
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      OpAsmParser::OperandType currentArgument, currentOperand;
      Type currentType;

      if (parser.parseRegionArgument(currentArgument) || parser.parseEqual() ||
          parser.parseOperand(currentOperand) ||
          parser.parseColonType(currentType))
        return failure();

      arguments.push_back(currentArgument);
      operands.push_back(currentOperand);
      operandTypes.push_back(currentType);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  // Parse the result types and the optional attributes
  SmallVector<Type, 8> resultTypes;
  if (parser.parseArrowTypeList(resultTypes) ||
      parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  // Resolve the operand types
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperands(operands, operandTypes, loc, state.operands) ||
      parser.addTypesToList(resultTypes, state.types))
    return failure();

  // Parse the body region.
  Region *body = state.addRegion();
  if (parser.parseRegion(*body, arguments, operandTypes))
    return failure();

  // Parse the optional bounds
  ArrayAttr lbAttr, ubAttr;
  if (succeeded(parser.parseOptionalKeyword("to"))) {
    // Parse the optional bounds
    if (parser.parseLParen() ||
        parser.parseAttribute(lbAttr, stencil::ApplyOp::getLBAttrName(),
                              state.attributes) ||
        parser.parseColon() ||
        parser.parseAttribute(ubAttr, stencil::ApplyOp::getUBAttrName(),
                              state.attributes) ||
        parser.parseRParen())
      return failure();
  }

  return success();
}

static void print(stencil::ApplyOp applyOp, OpAsmPrinter &printer) {
  printer << stencil::ApplyOp::getOperationName() << ' ';
  // Print the region arguments
  SmallVector<Value, 10> operands = applyOp.getOperands();
  if (!applyOp.region().empty() && !operands.empty()) {
    Block *body = applyOp.getBody();
    printer << "(";
    llvm::interleaveComma(
        llvm::seq<int>(0, operands.size()), printer, [&](int i) {
          printer << body->getArgument(i) << " = " << operands[i] << " : "
                  << operands[i].getType();
        });
    printer << ") ";
  }

  // Print the result types
  printer << "-> ";
  if (applyOp.res().size() > 1)
    printer << "(";
  llvm::interleaveComma(applyOp.res().getTypes(), printer);
  if (applyOp.res().size() > 1)
    printer << ")";

  // Print optional attributes
  printer.printOptionalAttrDictWithKeyword(
      applyOp.getAttrs(), /*elidedAttrs=*/{stencil::ApplyOp::getLBAttrName(),
                                           stencil::ApplyOp::getUBAttrName()});

  // Print region, bounds, and return type
  printer.printRegion(applyOp.region(),
                      /*printEntryBlockArgs=*/false);
  if (applyOp.lb().hasValue() && applyOp.ub().hasValue()) {
    printer << " to (";
    printer.printAttribute(applyOp.lb().getValue());
    printer << " : ";
    printer.printAttribute(applyOp.ub().getValue());
    printer << ")";
  }
}

void stencil::ApplyOp::updateArgumentTypes() {
  for (auto en : llvm::enumerate(getOperandTypes())) {
    if (en.value() != getBody()->getArgument(en.index()).getType()) {
      auto newType = en.value().cast<TempType>();
      auto oldType =
          getBody()->getArgument(en.index()).getType().cast<TempType>();
      // Check both are temporary and only the size changes
      assert(oldType.getElementType() == newType.getElementType() &&
             "expected the same element type");
      assert(oldType.getAllocation() == newType.getAllocation() &&
             "expected the same allocation");
      getBody()->getArgument(en.index()).setType(newType);
    }
  }
}

bool stencil::ApplyOp::hasOnlyEmptyStores() {
  auto result = walk([&](stencil::StoreResultOp resultOp) {
    if (resultOp.operands().size() != 0)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

//===----------------------------------------------------------------------===//
// stencil.dyn_access
//===----------------------------------------------------------------------===//

void stencil::DynAccessOp::shiftByOffset(ArrayRef<int64_t> offset) {
  // Compute the shifted extent
  Index lb, ub;
  std::tie(lb, ub) = getAccessExtent();
  lb = applyFunElementWise(offset, lb, std::plus<int64_t>());
  ub = applyFunElementWise(offset, ub, std::plus<int64_t>());
  // Create the attributes
  SmallVector<Attribute, kIndexSize> lbAttrs;
  SmallVector<Attribute, kIndexSize> ubAttrs;
  llvm::transform(lb, std::back_inserter(lbAttrs), [&](int64_t x) {
    return IntegerAttr::get(IntegerType::get(getContext(), 64), x);
  });
  llvm::transform(ub, std::back_inserter(ubAttrs), [&](int64_t x) {
    return IntegerAttr::get(IntegerType::get(getContext(), 64), x);
  });
  lbAttr(ArrayAttr::get(lbAttrs, getContext()));
  ubAttr(ArrayAttr::get(ubAttrs, getContext()));
}

std::tuple<stencil::Index, stencil::Index>
stencil::DynAccessOp::getAccessExtent() {
  Index lowerBound, upperBound;
  for (auto it : llvm::zip(lb(), ub())) {
    lowerBound.push_back(
        std::get<0>(it).cast<IntegerAttr>().getValue().getSExtValue());
    upperBound.push_back(
        std::get<1>(it).cast<IntegerAttr>().getValue().getSExtValue());
  }
  return std::make_tuple(lowerBound, upperBound);
}

//===----------------------------------------------------------------------===//
// stencil.store_result
//===----------------------------------------------------------------------===//

Optional<SmallVector<OpOperand *, 10>>
stencil::StoreResultOp::getReturnOpOperands() {
  // Keep a list of consumer operands and operations
  DenseSet<Operation *> currOperations;
  SmallVector<OpOperand *, 10> currOperands;
  for (auto &use : getResult().getUses()) {
    currOperands.push_back(&use);
    currOperations.insert(use.getOwner());
  }

  while (currOperations.size() == 1) {
    // Return the results of the return operation
    if (auto returnOp = dyn_cast<stencil::ReturnOp>(*currOperations.begin())) {
      return currOperands;
    }
    // Search the parent block for a return operation
    if (auto yieldOp = dyn_cast<scf::YieldOp>(*currOperations.begin())) {
      // Expected for ops in apply ops not to return a result
      if (isa<scf::ForOp>(yieldOp->getParentOp()) &&
          yieldOp->getParentOfType<stencil::ApplyOp>())
        return llvm::None;

      // Search the uses of the result and compute the consumer operations
      currOperations.clear();
      SmallVector<OpOperand *, 10> nextOperands;
      for (auto &use : currOperands) {
        auto result =
            yieldOp->getParentOp()->getResult(use->getOperandNumber());
        for (auto &use : result.getUses()) {
          nextOperands.push_back(&use);
          currOperations.insert(use.getOwner());
        }
      }
      currOperands.swap(nextOperands);
    } else {
      // Expected a return or a yield operation
      return llvm::None;
    }
  }
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// stencil.combine
//===----------------------------------------------------------------------===//

namespace {
// Check if operands connect one-by-one to one combine or to multiple apply ops
bool checkOneByOneOperandMapping(OperandRange base, OperandRange extra,
                                 ArrayRef<Operation *> definingOps) {
  // Check the defining op is a unique combine op with one-by-one mapping
  if (auto combineOp = dyn_cast<stencil::CombineOp>(*definingOps.begin())) {
    // Check all operands have one use
    if (!(llvm::all_of(base, [](Value value) { return value.hasOneUse(); }) &&
          llvm::all_of(extra, [](Value value) { return value.hasOneUse(); })))
      return false;
    return definingOps.size() == 1 &&
           combineOp.getNumResults() == base.size() + extra.size();
  }
  // Check the defining ops are apply ops with a one-by-one mapping
  unsigned numResults = 0;
  for (auto definingOp : definingOps) {
    // Check all defining ops are apply ops
    if (!isa<stencil::ApplyOp>(definingOp))
      return false;
    // Check the apply ops connect to combine ops only
    if (llvm::any_of(definingOp->getUsers(), [](Operation *op) {
          return !isa<stencil::CombineOp>(op);
        }))
      return false;
    numResults += definingOp->getNumResults();
  }
  // Check all operands are unique
  DenseSet<Value> operands;
  operands.insert(base.begin(), base.end());
  operands.insert(extra.begin(), extra.end());
  return numResults == operands.size() &&
         numResults == base.size() + extra.size();
}

// Helper to check type compatibility given the combine dim
bool checkTempTypesMatch(Type type1, Type type2, unsigned dim) {
  auto tempType1 = type1.cast<TempType>();
  auto tempType2 = type2.cast<TempType>();
  // Check the element type
  if (tempType1.getElementType() != tempType2.getElementType())
    return false;
  // Check the shape of static shapes match
  for (auto en : llvm::enumerate(tempType1.getShape())) {
    // Skip the combine dim
    if (en.index() == dim)
      continue;
    // Check neither of the sizes is dynamic
    auto size1 = en.value();
    auto size2 = tempType2.getShape()[en.index()];
    if (GridType::isDynamic(size1) || GridType::isDynamic(size2))
      continue;
    // Check the sizes match
    if (size1 != size2)
      return false;
  }
  return true;
}
} // namespace

static LogicalResult verify(stencil::CombineOp op) {
  // Check the combine op has at least one operand
  if (op.getNumOperands() == 0)
    return op.emitOpError("expected the operand list to be non-empty");

  // Check the operand and result sizes match
  if (op.lower().size() != op.upper().size())
    return op.emitOpError("expected the lower and upper operand size to match");
  if (op.res().size() !=
      op.lower().size() + op.lowerext().size() + op.upperext().size())
    return op.emitOpError("expected the result and operand sizes to match");

  // Check all inputs have a defining op
  if (!llvm::all_of(op.getOperands(),
                    [](Value value) { return value.getDefiningOp(); }))
    return op.emitOpError("expected the operands to have a defining op");

  // Check the lower and upper operand types match
  if (!llvm::all_of(llvm::zip(op.lower().getTypes(), op.upper().getTypes()),
                    [&](std::tuple<Type, Type> x) {
                      return checkTempTypesMatch(std::get<0>(x), std::get<1>(x),
                                                 op.dim());
                    }))
    return op.emitOpError("expected lower and upper operand types to match");

  // Check the lower/upper operand types match the result types
  if (!llvm::all_of(llvm::zip(op.lower().getTypes(), op.res().getTypes()),
                    [&](std::tuple<Type, Type> x) {
                      return checkTempTypesMatch(std::get<0>(x), std::get<1>(x),
                                                 op.dim());
                    }))
    return op.emitOpError("expected the lower/upper and result types to match");

  // Check the if the extra types match the corresponding result types
  auto lowerExtResTypes = op.res().getTypes().drop_front(op.lower().size());
  auto upperExtResTypes = op.res().getTypes().take_back(op.upperext().size());
  if (!llvm::all_of(llvm::zip(op.lowerext().getTypes(), lowerExtResTypes),
                    [&](std::tuple<Type, Type> x) {
                      return checkTempTypesMatch(std::get<0>(x), std::get<1>(x),
                                                 op.dim());
                    }))
    return op.emitOpError("expected the lowerext and result types to match");
  if (!llvm::all_of(llvm::zip(op.upperext().getTypes(), upperExtResTypes),
                    [&](std::tuple<Type, Type> x) {
                      return checkTempTypesMatch(std::get<0>(x), std::get<1>(x),
                                                 op.dim());
                    }))
    return op.emitOpError("expected the upperext and result types to match");

  // Check the operands either connect to one combine or multiple apply ops
  auto lowerDefiningOps = op.getLowerDefiningOps();
  auto upperDefiningOps = op.getUpperDefiningOps();
  if (!checkOneByOneOperandMapping(op.lower(), op.lowerext(), lowerDefiningOps))
    return op.emitOpError("expected the lower operands to connect one-by-one "
                          "to one combine or multiple apply ops");
  if (!checkOneByOneOperandMapping(op.upper(), op.upperext(), upperDefiningOps))
    return op.emitOpError("expected the upper operands to connect one-by-one "
                          "to one combine or multiple apply ops");
  return success();
}

stencil::CombineOp stencil::CombineOp::getCombineTreeRoot() {
  auto curr = this->getOperation();
  while (std::distance(curr->getUsers().begin(), curr->getUsers().end()) == 1 &&
         llvm::all_of(curr->getUsers(), [](Operation *op) {
           return isa<stencil::CombineOp>(op);
         })) {
    curr = *curr->getUsers().begin();
  }
  return cast<stencil::CombineOp>(curr);
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

stencil::ApplyOpPattern::ApplyOpPattern(MLIRContext *context,
                                        PatternBenefit benefit)
    : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}

stencil::ApplyOp
stencil::ApplyOpPattern::cleanupOpArguments(stencil::ApplyOp applyOp,
                                            PatternRewriter &rewriter) const {
  // Compute the new operand list and index mapping
  llvm::DenseMap<Value, unsigned int> newIndex;
  SmallVector<Value, 10> newOperands;
  for (auto &en : llvm::enumerate(applyOp.getOperands())) {
    if (newIndex.count(en.value()) == 0) {
      if (!applyOp.getBody()->getArgument(en.index()).getUses().empty()) {
        newIndex[en.value()] = newOperands.size();
        newOperands.push_back(en.value());
      }
    }
  }

  // Create a new operation with shorther argument list
  if (newOperands.size() < applyOp.getNumOperands()) {
    auto loc = applyOp.getLoc();
    auto newOp = rewriter.create<stencil::ApplyOp>(
        loc, applyOp.getResultTypes(), newOperands, applyOp.lb(), applyOp.ub());

    // Compute the argument mapping and move the block
    SmallVector<Value, 10> newArgs(applyOp.getNumOperands());
    llvm::transform(applyOp.getOperands(), newArgs.begin(), [&](Value value) {
      return newIndex.count(value) == 0
                 ? nullptr // pass default value if the new apply has no params
                 : newOp.getBody()->getArgument(newIndex[value]);
    });
    rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(), newArgs);
    return newOp;
  }
  return nullptr;
}

stencil::CombineOpPattern::CombineOpPattern(MLIRContext *context,
                                            PatternBenefit benefit)
    : OpRewritePattern<stencil::CombineOp>(context, benefit) {}

stencil::ApplyOp stencil::CombineOpPattern::createEmptyApply(
    stencil::CombineOp combineOp, int64_t lowerLimit, int64_t upperLimit,
    ValueRange values, PatternRewriter &rewriter) const {
  // Get the location of the mirrored return operation
  auto loc = combineOp.getLoc();

  // Get the return op attached to the operand range
  auto applyOp = cast<stencil::ApplyOp>(values.front().getDefiningOp());
  auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());

  // Get the shape of the combine op
  auto shapeOp = cast<ShapeOp>(combineOp.getOperation());
  Index lb, ub;

  // Compute the result types depending on the size information
  SmallVector<Type, 10> newResultTypes;
  if (shapeOp.hasShape()) {
    // Compute the shape of the empty apply
    lb = shapeOp.getLB();
    ub = shapeOp.getUB();
    lb[combineOp.dim()] = max(lowerLimit, lb[combineOp.dim()]);
    ub[combineOp.dim()] = min(upperLimit, ub[combineOp.dim()]);

    // Resize the operand types
    for (auto value : values) {
      auto operandType = value.getType().cast<TempType>();
      auto shape = applyFunElementWise(ub, lb, std::minus<int64_t>());
      newResultTypes.push_back(
          TempType::get(operandType.getElementType(), shape));
    }
  } else {
    // Assume the types have a dynamic shape
    for (auto value : values) {
      auto operandType = value.getType().cast<TempType>();
      assert(operandType.hasDynamicShape() &&
             "expected operand type to have a dynamic shape");
    }
  }

  // Create an empty apply op including empty stores
  auto newOp = rewriter.create<stencil::ApplyOp>(
      returnOp.getLoc(), newResultTypes,
      lb.empty() ? nullptr : rewriter.getI64ArrayAttr(lb),
      ub.empty() ? nullptr : rewriter.getI64ArrayAttr(ub));
  newOp.region().push_back(new Block());

  // Update the body of the apply op
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newOp.getBody());

  // Create the empty stores and the return op
  SmallVector<Value, 10> newOperands;
  for (auto newResultType : newResultTypes) {
    auto elementType = newResultType.cast<TempType>().getElementType();
    auto resultOp = rewriter.create<stencil::StoreResultOp>(
        loc, ResultType::get(elementType), ValueRange());
    newOperands.append(returnOp.getUnrollFac(), resultOp);
  }
  rewriter.create<stencil::ReturnOp>(loc, newOperands, returnOp.unroll());
  return newOp;
}

namespace {

/// This is a pattern to remove duplicate loads
struct ApplyOpLoadCleaner : public stencil::ApplyOpPattern {
  using ApplyOpPattern::ApplyOpPattern;

  LogicalResult cleanupLoadOps(DenseSet<Operation *> &loadOps,
                               stencil::ApplyOp applyOp,
                               PatternRewriter &rewriter) const {
    // Check all load ops have a shape (otherwise cse is sufficient)
    if (llvm::any_of(loadOps,
                     [](ShapeOp shapeOp) { return !shapeOp.hasShape(); }))
      return failure();

    // Compute the bounding box of all load shapes
    auto lb = cast<ShapeOp>(*loadOps.begin()).getLB();
    auto ub = cast<ShapeOp>(*loadOps.begin()).getUB();
    for (auto loadOp : loadOps) {
      auto shapeOp = cast<ShapeOp>(loadOp);
      lb = applyFunElementWise(shapeOp.getLB(), lb, min);
      ub = applyFunElementWise(shapeOp.getUB(), ub, max);
    }

    // Create a new load operation
    auto loadOp = rewriter.create<stencil::LoadOp>(
        applyOp.getLoc(), cast<stencil::LoadOp>(*loadOps.begin()).field(),
        rewriter.getI64ArrayAttr(lb), rewriter.getI64ArrayAttr(ub));

    // Compute the new operand list
    SmallVector<Value, 10> newOperands;
    llvm::transform(applyOp.getOperands(), std::back_inserter(newOperands),
                    [&](Value value) {
                      return loadOps.count(value.getDefiningOp()) == 1 ? loadOp
                                                                       : value;
                    });

    // Replace the apply operation using the new load op
    auto newOp = rewriter.create<stencil::ApplyOp>(
        applyOp.getLoc(), applyOp.getResultTypes(), newOperands, applyOp.lb(),
        applyOp.ub());
    rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(),
                         newOp.getBody()->getArguments());
    rewriter.replaceOp(applyOp, newOp.getResults());
    return success();
  }

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Compute mapping of the loaded fields to the load ops
    DenseMap<Value, DenseSet<Operation *>> fieldToLoadOps;
    for (auto value : applyOp.getOperands()) {
      if (auto loadOp =
              dyn_cast_or_null<stencil::LoadOp>(value.getDefiningOp())) {
        fieldToLoadOps[loadOp.field()].insert(loadOp.getOperation());
      }
    }
    // Replace multiple loads of the same field
    for (auto entry : fieldToLoadOps) {
      if (entry.getSecond().size() > 1) {
        return cleanupLoadOps(entry.getSecond(), applyOp, rewriter);
      }
    }
    return failure();
  }
};

/// This is a pattern to remove duplicate and unused arguments
struct ApplyOpArgCleaner : public stencil::ApplyOpPattern {
  using ApplyOpPattern::ApplyOpPattern;

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    if (auto newOp = cleanupOpArguments(applyOp, rewriter)) {
      rewriter.replaceOp(applyOp, newOp.getResults());
      return success();
    }
    return failure();
  }
};

/// This is a pattern removes unused results
struct ApplyOpResCleaner : public stencil::ApplyOpPattern {
  using ApplyOpPattern::ApplyOpPattern;

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Compute the updated result list
    SmallVector<OpResult, 10> usedResults;
    llvm::copy_if(applyOp.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    if (usedResults.size() != applyOp.getNumResults()) {
      // Erase the op if it has not uses
      if (usedResults.size() == 0) {
        rewriter.eraseOp(applyOp);
        return success();
      }

      // Get the return operation
      auto returnOp =
          cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
      unsigned unrollFac = returnOp.getUnrollFac();

      // Compute the new result and and return op operand vector
      SmallVector<Type, 10> newResultTypes;
      SmallVector<Value, 10> newOperands;
      for (auto usedResult : usedResults) {
        newResultTypes.push_back(usedResult.getType());
        auto slice = returnOp.getOperands().slice(
            usedResult.getResultNumber() * unrollFac, unrollFac);
        newOperands.append(slice.begin(), slice.end());
      }

      // Create a new apply operation
      auto newOp = rewriter.create<stencil::ApplyOp>(
          applyOp.getLoc(), newResultTypes, applyOp.getOperands(), applyOp.lb(),
          applyOp.ub());
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<stencil::ReturnOp>(returnOp.getLoc(), newOperands,
                                         returnOp.unroll());
      rewriter.eraseOp(returnOp);
      rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(),
                           newOp.getBody()->getArguments());

      // Compute the replacement results
      SmallVector<Value, 10> repResults(applyOp.getNumResults(),
                                        newOp.getResults().front());
      for (auto en : llvm::enumerate(usedResults))
        repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
      rewriter.replaceOp(applyOp, repResults);
      return success();
    }
    return failure();
  }
};

/// This is a pattern to removes combines with symmetric operands
struct CombineOpSymmetricCleaner : public stencil::CombineOpPattern {
  using CombineOpPattern::CombineOpPattern;

  stencil::ApplyOp getDefiningApplyOp(Value value) const {
    return dyn_cast_or_null<stencil::ApplyOp>(value.getDefiningOp());
  }

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Exit if the combine has extra operands
    if (combineOp.lowerext().size() > 0 || combineOp.upperext().size() > 0)
      return failure();

    // Compute the empty values
    SmallVector<Value, 10> emptyValues;
    for (auto en : llvm::enumerate(combineOp.lower())) {
      auto lowerOp = getDefiningApplyOp(en.value());
      auto upperOp = getDefiningApplyOp(combineOp.upper()[en.index()]);
      if (lowerOp && upperOp && lowerOp.hasOnlyEmptyStores() &&
          upperOp.hasOnlyEmptyStores()) {
        emptyValues.push_back(en.value());
      }
    }

    // Compare the upper and lower values
    for (auto en : llvm::enumerate(combineOp.lower())) {
      if (en.value() != combineOp.upper()[en.index()] &&
          !llvm::is_contained(emptyValues, en.value()))
        return failure();
    }

    // Create an empty apply
    ApplyOp emptyOp;
    if (!emptyValues.empty()) {
      emptyOp = createEmptyApply(combineOp, std::numeric_limits<int64_t>::min(),
                                 std::numeric_limits<int64_t>::max(),
                                 emptyValues, rewriter);
    }

    // Compute the replacement values
    unsigned emptyCount = 0;
    SmallVector<Value, 10> repResults;
    for (auto en : llvm::enumerate(combineOp.lower())) {
      if (en.value() == combineOp.upper()[en.index()]) {
        repResults.push_back(en.value());
      } else {
        repResults.push_back(emptyOp.getResult(emptyCount++));
      }
    }

    // Replace the combine op
    rewriter.replaceOp(combineOp, repResults);
    return success();
  }
};

/// This is a pattern to remove combines that do not split the domain
struct CombineOpEmptyCleaner : public stencil::CombineOpPattern {
  using CombineOpPattern::CombineOpPattern;

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Check if the index of the combine op is inside the shape
    auto shapeOp = cast<ShapeOp>(combineOp.getOperation());
    if (shapeOp.hasShape()) {
      // Remove the upper operands if the index is larger than the upper bound
      if (combineOp.getIndex() > shapeOp.getUB()[combineOp.dim()]) {
        // Compute the replacement results
        SmallVector<Value, 10> repResults = combineOp.lower();
        repResults.append(combineOp.lowerext().begin(),
                          combineOp.lowerext().end());

        // Introduce empty stores in case there are upper extra results
        if (combineOp.upperext().size() > 0) {
          auto newOp =
              createEmptyApply(combineOp, std::numeric_limits<int64_t>::min(),
                               std::numeric_limits<int64_t>::max(),
                               combineOp.upperext(), rewriter);
          repResults.append(newOp.getResults().begin(),
                            newOp.getResults().end());
        }

        // Replace the combine op
        rewriter.replaceOp(combineOp, repResults);
        return success();
      }
      // Remove the lower operands if the index is smaller than the lower bound
      if (combineOp.getIndex() < shapeOp.getLB()[combineOp.dim()]) {
        // Compute the replacement results
        SmallVector<Value, 10> repResults = combineOp.upper();

        // Introduce empty stores in case there are lower extra results
        if (combineOp.lowerext().size() > 0) {
          auto newOp =
              createEmptyApply(combineOp, std::numeric_limits<int64_t>::min(),
                               std::numeric_limits<int64_t>::max(),
                               combineOp.lowerext(), rewriter);
          repResults.append(newOp.getResults().begin(),
                            newOp.getResults().end());
        }
        repResults.append(combineOp.upperext().begin(),
                          combineOp.upperext().end());

        // Replace the combine op
        rewriter.replaceOp(combineOp, repResults);
        return success();
      }
    }
    return failure();
  }
};

/// This is a pattern to remove unused arguments
struct CombineOpResCleaner : public stencil::CombineOpPattern {
  using CombineOpPattern::CombineOpPattern;

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Compute the updated result list
    SmallVector<OpResult, 10> usedResults;
    llvm::copy_if(combineOp.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    if (usedResults.size() != combineOp.getNumResults()) {
      // Erase the op if it has not uses
      if (usedResults.size() == 0) {
        rewriter.eraseOp(combineOp);
        return success();
      }

      // Compute the new result types and operands
      SmallVector<Type, 10> newResultTypes;
      llvm::transform(usedResults, std::back_inserter(newResultTypes),
                      [](Value value) { return value.getType(); });

      SmallVector<Value, 10> newLowerOperands, newLowerExtraOperands;
      SmallVector<Value, 10> newUpperOperands, newUpperExtraOperands;
      for (auto used : usedResults) {
        unsigned resultNumber = used.getResultNumber();
        // Copy the main operands
        if (auto num = combineOp.getLowerOperandNumber(resultNumber)) {
          newLowerOperands.push_back(combineOp.lower()[num.getValue()]);
          newUpperOperands.push_back(combineOp.upper()[num.getValue()]);
        }
        // Copy the lower extra operands
        if (auto num = combineOp.getLowerExtraOperandNumber(resultNumber)) {
          newLowerExtraOperands.push_back(combineOp.lowerext()[num.getValue()]);
        }
        // Copy the upper extra operands
        if (auto num = combineOp.getUpperExtraOperandNumber(resultNumber)) {
          newUpperExtraOperands.push_back(combineOp.upperext()[num.getValue()]);
        }
      }

      // Create a new combine op that returns only the used results
      auto newOp = rewriter.create<stencil::CombineOp>(
          combineOp.getLoc(), newResultTypes, combineOp.dim(),
          combineOp.getIndex(), newLowerOperands, newUpperOperands,
          newLowerExtraOperands, newUpperExtraOperands, combineOp.lbAttr(),
          combineOp.ubAttr());

      // Compute the replacement results
      SmallVector<Value, 10> repResults(combineOp.getNumResults(),
                                        newOp.getResults().front());
      for (auto en : llvm::enumerate(usedResults))
        repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
      rewriter.replaceOp(combineOp, repResults);
      return success();
    }
    return failure();
  }
};

// Helper methods to hoist operations
LogicalResult hoistBackward(Operation *op, PatternRewriter &rewriter,
                            std::function<bool(Operation *)> condition) {
  // Skip compute operations
  auto curr = op;
  while (curr->getPrevNode() && condition(curr->getPrevNode()) &&
         !llvm::is_contained(curr->getPrevNode()->getUsers(), op))
    curr = curr->getPrevNode();

  // Move the operation
  if (curr != op) {
    rewriter.setInsertionPoint(curr);
    rewriter.replaceOp(op, rewriter.clone(*op)->getResults());
    return success();
  }
  return failure();
}
LogicalResult hoistForward(Operation *op, PatternRewriter &rewriter,
                           std::function<bool(Operation *)> condition) {
  // Skip compute operations
  auto curr = op;
  while (curr->getNextNode() && condition(curr->getNextNode()) &&
         !curr->getNextNode()->isKnownTerminator())
    curr = curr->getNextNode();

  // Move the operation
  if (curr != op) {
    rewriter.setInsertionPointAfter(curr);
    rewriter.replaceOp(op, rewriter.clone(*op)->getResults());
    return success();
  }
  return failure();
} // namespace

/// This is a pattern to hoist assert ops out of the computation
struct CastOpHoisting : public OpRewritePattern<stencil::CastOp> {
  using OpRewritePattern<stencil::CastOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    // Skip all operations except for other casts
    auto condition = [](Operation *op) { return !isa<stencil::CastOp>(op); };
    return hoistBackward(castOp.getOperation(), rewriter, condition);
  }
};

/// This is a pattern to hoist load ops out of the computation
struct LoadOpHoisting : public OpRewritePattern<stencil::LoadOp> {
  using OpRewritePattern<stencil::LoadOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Skip all operations except for casts and other loads
    auto condition = [](Operation *op) {
      return !isa<stencil::LoadOp>(op) && !isa<stencil::CastOp>(op);
    };
    return hoistBackward(loadOp.getOperation(), rewriter, condition);
  }
};

/// This is a pattern to hoist store ops out of the computation
struct StoreOpHoisting : public OpRewritePattern<stencil::StoreOp> {
  using OpRewritePattern<stencil::StoreOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Skip all operations except for stores
    auto condition = [](Operation *op) { return !isa<stencil::StoreOp>(op); };
    return hoistForward(storeOp.getOperation(), rewriter, condition);
  }
};

} // end anonymous namespace

// Register canonicalization patterns
void stencil::ApplyOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ApplyOpArgCleaner, ApplyOpResCleaner, ApplyOpLoadCleaner>(
      context);
}

void stencil::CombineOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CombineOpResCleaner, CombineOpEmptyCleaner,
                 CombineOpSymmetricCleaner>(context);
}

void stencil::CastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CastOpHoisting>(context);
}
void stencil::LoadOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<LoadOpHoisting>(context);
}
void stencil::StoreOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<StoreOpHoisting>(context);
}

namespace mlir {
namespace stencil {
#include "Dialect/Stencil/StencilOpsInterfaces.cpp.inc"
} // namespace stencil
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.cpp.inc"
