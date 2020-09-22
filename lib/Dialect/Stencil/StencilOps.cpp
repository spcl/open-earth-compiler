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
#include <algorithm>
#include <bits/stdint-intn.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>

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
    return IntegerAttr::get(IntegerType::get(64, getContext()), x);
  });
  llvm::transform(ub, std::back_inserter(ubAttrs), [&](int64_t x) {
    return IntegerAttr::get(IntegerType::get(64, getContext()), x);
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
// stencil.make_result
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
      if (isa<scf::ForOp>(yieldOp.getParentOp()) &&
          yieldOp.getParentOfType<stencil::ApplyOp>())
        return llvm::None;

      // Search the uses of the result and compute the consumer operations
      currOperations.clear();
      SmallVector<OpOperand *, 10> nextOperands;
      for (auto &use : currOperands) {
        auto result = yieldOp.getParentOp()->getResult(use->getOperandNumber());
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
// Helper function to check if operands connect one-by-one to the same combine
bool checkIfOneByOneMappingToCombineOp(OperandRange operands) {
  DenseSet<Operation *> definingOps;
  for (auto operand : operands)
    definingOps.insert(operand.getDefiningOp());
  // Check if the operation is unique
  if (definingOps.size() != 1)
    return false;
  // Check all operands have one use
  if (!llvm::all_of(operands, [](Value value) { return value.hasOneUse(); }))
    return false;
  // Check the operation is a combine op with a one-by-one mapping
  if (auto combineOp = dyn_cast<stencil::CombineOp>(*definingOps.begin()))
    return combineOp.getNumResults() == operands.size();
  return false;
}

// Helper function to check if all operands are connected to apply ops
bool checkIfConnectedToApplyOps(OperandRange operands) {
  return llvm::all_of(operands, [](Value value) {
    return isa<stencil::ApplyOp>(value.getDefiningOp());
  });
}
} // namespace

static LogicalResult verify(stencil::CombineOp op) {
  // Check the combine op has at least one operand
  if (op.getNumOperands() == 0)
    return op.emitOpError("expected the operand list to be non-empty");

  // Check the operand and result sizes match
  if (op.lower().size() != op.upper().size() ||
      op.lower().size() != op.res().size())
    return op.emitOpError("expected the operand and result sizes to match");

  // Check the operand and result types match
  if (!llvm::all_of(llvm::zip(op.lower().getTypes(), op.upper().getTypes(),
                              op.res().getTypes()),
                    [&](std::tuple<Type, Type, Type> x) {
                      SmallVector<TempType, 3> tempTypes = {
                          std::get<0>(x).cast<TempType>(),
                          std::get<1>(x).cast<TempType>(),
                          std::get<2>(x).cast<TempType>()};
                      // Check the element types match
                      if (!llvm::all_of(tempTypes, [&](TempType type) {
                            return type.getElementType() ==
                                   tempTypes.front().getElementType();
                          }))
                        return false;
                      // Check the shapes match except for the combine dim
                      if (llvm::any_of(tempTypes, [&](TempType type) {
                            for (auto en : llvm::enumerate(type.getShape())) {
                              if (en.index() != op.dim() &&
                                  en.value() !=
                                      tempTypes.front().getShape()[en.index()])
                                return true;
                            }
                            return false;
                          }))
                        return false;
                      // Otherwise the types match
                      return true;
                    }))
    return op.emitOpError("expected the operand and result types to match");

  // Check all inputs have a defining op
  if (!llvm::all_of(op.getOperands(),
                    [](Value value) { return value.getDefiningOp(); }))
    return op.emitOpError("expected the operands to have a defining op");

  // Check the operands either connect to one combine or multiple apply ops
  if (!(checkIfOneByOneMappingToCombineOp(op.lower()) ||
        checkIfConnectedToApplyOps(op.lower())))
    return op.emitOpError("expected the lower operands to be defined by apply "
                          "ops or by a single combine op");
  if (!(checkIfOneByOneMappingToCombineOp(op.upper()) ||
        checkIfConnectedToApplyOps(op.upper())))
    return op.emitOpError("expected the lower operands to be defined by apply "
                          "ops or by a single combine op");
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
      } else {
        // Unused arguments are mapped to the first index
        newIndex[en.value()] = 0;
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
      return newOperands.empty()
                 ? value // pass default value if the new apply has no params
                 : newOp.getBody()->getArgument(newIndex[value]);
    });
    rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(), newArgs);
    return newOp;
  }
  return nullptr;
}

namespace {

/// This is a pattern to remove duplicate and unused arguments
struct ApplyOpArgumentCleaner : public stencil::ApplyOpPattern {
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
struct ApplyOpResultCleaner : public stencil::ApplyOpPattern {
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
      for (auto en : llvm::enumerate(usedResults)) {
        repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
      }
      rewriter.replaceOp(applyOp, repResults);
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
  results.insert<ApplyOpArgumentCleaner, ApplyOpResultCleaner>(context);
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

#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.cpp.inc"

} // namespace stencil
} // namespace mlir
