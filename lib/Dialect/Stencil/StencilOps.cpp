#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <bits/stdint-intn.h>
#include <cstddef>
#include <functional>
#include <iterator>

using namespace mlir;

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

void stencil::ApplyOp::setOperandShape(Value operand, TempType newType) {
  auto it = llvm::find(getOperands(), operand);
  assert(it != getOperands().end() && "failed to find operand");
  auto arg = getBody()->getArgument(std::distance(getOperands().begin(), it));
  auto oldType = arg.getType().cast<TempType>();
  assert(oldType.getElementType() == newType.getElementType() &&
         "expected the types to have the same element type");
  assert(oldType.getAllocation() == newType.getAllocation() &&
         "expected the types to have the same allocation");
  arg.setType(newType);
}

namespace {

/// This is a pattern to remove duplicate results
struct ApplyOpResCleaner : public OpRewritePattern<stencil::ApplyOp> {
  using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Compute the new return op operands
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    SmallVector<Value, 10> newOperands;
    for (unsigned i = 0, e = returnOp.getNumOperands(); i != e; ++i) {
      if (!llvm::is_contained(newOperands, returnOp.getOperand(i)))
        newOperands.push_back(returnOp.getOperand(i));
    }

    // Remove duplicates if needed
    if (newOperands.size() < returnOp.getNumOperands()) {
      // Replace the return op
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<stencil::ReturnOp>(returnOp.getLoc(), newOperands,
                                         returnOp.unroll());
      // Compute the apply op results
      SmallVector<Value, 10> newResults;
      SmallVector<size_t, 10> newResultIndexes;
      unsigned factor = returnOp.getUnrollFactor();
      for (unsigned i = 0, e = applyOp.getNumResults(); i != e; ++i) {
        auto it = llvm::find(newOperands, returnOp.getOperand(i * factor));
        size_t index = std::distance(newOperands.begin(), it);
        assert(index % factor == 0 && "expected multiple of unroll factor");
        assert(index / factor <= i && "expected lower replacement index");
        // Add new results
        if (index == i * factor)
          newResults.push_back(applyOp.getResult(i));
        newResultIndexes.push_back(index / factor);
      }

      // Clone the apply op
      rewriter.setInsertionPoint(applyOp);
      auto newOp = rewriter.create<stencil::ApplyOp>(
          applyOp.getLoc(), applyOp.getOperands(), newResults);
      rewriter.inlineRegionBefore(applyOp.region(), newOp.region(),
                                  newOp.region().begin());

      // Replace all uses of the applyOp results
      SmallVector<Value, 10> repResults;
      for (size_t i = 0, e = applyOp.getNumResults(); i != e; ++i) {
        repResults.push_back(newOp.getResult(newResultIndexes[i]));
      }
      rewriter.replaceOp(applyOp, repResults);
      rewriter.eraseOp(returnOp);
      return success();
    }
    return failure();
  }
};

/// This is a pattern to remove duplicate arguments
struct ApplyOpArgCleaner : public OpRewritePattern<stencil::ApplyOp> {
  using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Compute the new Operand list
    SmallVector<Value, 10> newOperands;
    for (unsigned i = 0, e = applyOp.getNumOperands(); i != e; ++i) {
      if (!llvm::is_contained(newOperands, applyOp.getOperand(i)))
        newOperands.push_back(applyOp.getOperand(i));
    }

    if (newOperands.size() < applyOp.getNumOperands()) {
      // Clone the apply op
      auto loc = applyOp.getLoc();
      auto newOp = rewriter.create<stencil::ApplyOp>(loc, newOperands,
                                                     applyOp.getResults());

      // Compute the block argument mapping
      BlockAndValueMapping mapper;
      for (unsigned i = 0, e = applyOp.getNumOperands(); i != e; ++i) {
        auto it = llvm::find(newOperands, applyOp.getOperand(i));
        size_t index = std::distance(newOperands.begin(), it);
        assert(index <= i && "expected lower replacement index");
        mapper.map(applyOp.getBody()->getArgument(i),
                   newOp.getBody()->getArgument(index));
      }

      // Clone the body
      rewriter.setInsertionPointToStart(newOp.getBody());
      for (auto &op : applyOp.getBody()->getOperations()) {
        rewriter.clone(op, mapper);
      }

      // Replace all uses of the applyOp results
      rewriter.replaceOp(applyOp, newOp.getResults());
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
  while (curr->getPrevNode() && condition(curr->getPrevNode())) {
    curr = curr->getPrevNode();
  }

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
  while (curr->getNextNode() && condition(curr->getNextNode())) {
    curr = curr->getNextNode();
  }

  // Move the operation
  if (curr != op) {
    rewriter.setInsertionPointAfter(curr);
    rewriter.replaceOp(op, rewriter.clone(*op)->getResults());
    return success();
  }
  return failure();
}

/// This is a pattern to hoist assert ops out of the computation
struct AssertOpHoisting : public OpRewritePattern<stencil::AssertOp> {
  using OpRewritePattern<stencil::AssertOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::AssertOp assertOp,
                                PatternRewriter &rewriter) const override {
    // Skip all non assert operations
    auto condition = [](Operation *op) { return !isa<stencil::AssertOp>(op); };
    return hoistBackward(assertOp.getOperation(), rewriter, condition);
  }
};

/// This is a pattern to hoist load ops out of the computation
struct LoadOpHoisting : public OpRewritePattern<stencil::LoadOp> {
  using OpRewritePattern<stencil::LoadOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Skip all apply and store operations
    auto condition = [](Operation *op) {
      return isa<stencil::ApplyOp>(op) || isa<stencil::StoreOp>(op);
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
    // Skip all apply and load operations
    auto condition = [](Operation *op) {
      return isa<stencil::ApplyOp>(op) || isa<stencil::LoadOp>(op);
    };
    return hoistForward(storeOp.getOperation(), rewriter, condition);
  }
};

} // end anonymous namespace

// Register canonicalization patterns
void stencil::ApplyOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ApplyOpArgCleaner, ApplyOpResCleaner>(context);
}

void stencil::AssertOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<AssertOpHoisting>(context);
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
