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
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <functional>
#include <iterator>

using namespace mlir;

//===----------------------------------------------------------------------===//
// stencil.assert
//===----------------------------------------------------------------------===//

void stencil::AssertOp::build(OpBuilder &builder, OperationState &state,
                              Value field, ArrayRef<int64_t> lb,
                              ArrayRef<int64_t> ub) {
  // Make sure that the offset has the right size
  assert(lb.size() == stencil::kNumOfDimensions &&
         ub.size() == stencil::kNumOfDimensions &&
         "expected bounds to have an element for every dimension");

  // Add an SSA arguments
  state.addOperands({field});
  // Add the bounds attributes
  state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
  state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub));
}

static ParseResult parseAssertOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType field;
  ArrayAttr lbAttr, ubAttr;
  Type fieldType;

  // Parse the assert op
  if (parser.parseOperand(field) || parser.parseLParen() ||
      parser.parseAttribute(lbAttr, stencil::AssertOp::getLBAttrName(),
                            state.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(ubAttr, stencil::AssertOp::getUBAttrName(),
                            state.attributes) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType(fieldType))
    return failure();

  // Make sure bounds have the right number of dimensions
  if (lbAttr.size() != stencil::kNumOfDimensions ||
      ubAttr.size() != stencil::kNumOfDimensions) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected bounds to have a component for every dimension");
    return failure();
  }

  if (parser.resolveOperand(field, fieldType, state.operands))
    return failure();

  return success();
}

static void print(stencil::AssertOp assertOp, OpAsmPrinter &printer) {
  Value field = assertOp.field();
  ArrayAttr lb = assertOp.lb();
  ArrayAttr ub = assertOp.ub();

  printer << stencil::AssertOp::getOperationName();
  printer << " " << field << " (";
  printer.printAttribute(lb);
  printer << ":";
  printer.printAttribute(ub);
  printer << ")";
  printer.printOptionalAttrDict(assertOp.getAttrs(), /*elidedAttrs=*/{
                                    stencil::AssertOp::getLBAttrName(),
                                    stencil::AssertOp::getUBAttrName()});
  printer << " : ";
  printer.printType(assertOp.field().getType());
}

static LogicalResult verify(stencil::AssertOp assertOp) {
  // Check if all uses are loads or stores
  int stores = 0;
  int loads = 0;
  int asserts = 0;
  for (OpOperand &use : assertOp.field().getUses()) {
    if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner()))
      stores++;
    if (auto loadOp = dyn_cast<stencil::LoadOp>(use.getOwner()))
      loads++;
    if (auto assertOp = dyn_cast<stencil::AssertOp>(use.getOwner()))
      asserts++;
  }
  // Check if input and output
  if (loads > 0 && stores > 0)
    return assertOp.emitOpError("field cannot by input and output");
  // Check if multiple stores
  if (stores > 1)
    return assertOp.emitOpError("field written multiple times");
  // Check if multiple asserts
  if (asserts != 1)
    return assertOp.emitOpError("multiple asserts for the same field");

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.access
//===----------------------------------------------------------------------===//

void stencil::AccessOp::build(OpBuilder &builder, OperationState &state,
                              Value temp, ArrayRef<int64_t> offset) {
  // Make sure that the offset has the right size
  assert(offset.size() == stencil::kNumOfDimensions &&
         "expected offset to have an element for every dimension");

  // Extract the element type of the temp.
  Type elementType = temp.getType().cast<stencil::TempType>().getElementType();

  // Add an SSA argument
  state.addOperands(temp);
  // Add the offset attribute
  state.addAttribute(getOffsetAttrName(), builder.getI64ArrayAttr(offset));
  // Set the return type
  state.addTypes(elementType);
}

static ParseResult parseAccessOp(OpAsmParser &parser, OperationState &state) {
  FunctionType funcType;
  ArrayAttr offset;
  SmallVector<OpAsmParser::OperandType, 1> operands;

  // Parse the temp
  if (parser.parseOperandList(operands) ||
      parser.parseAttribute(offset, stencil::AccessOp::getOffsetAttrName(),
                            state.attributes))
    return failure();
  // Make sure it has the right number of dimensions
  if (offset.size() != stencil::kNumOfDimensions) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected offset to have a component for every dimension");
    return failure();
  }

  // Parse optional attributes as well as the temp type
  if (parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType<FunctionType>(funcType) ||
      parser.resolveOperands(operands, funcType.getInputs(),
                             parser.getCurrentLocation(), state.operands))
    return failure();

  // Add the return value
  if (parser.addTypesToList(funcType.getResults(), state.types))
    return failure();

  return success();
}

static void print(stencil::AccessOp accessOp, OpAsmPrinter &printer) {
  Value temp = accessOp.temp();
  ArrayAttr offset = accessOp.offset();

  printer << stencil::AccessOp::getOperationName() << ' ' << temp;
  printer.printAttribute(offset);
  printer.printOptionalAttrDict(accessOp.getAttrs(), /*elidedAttrs=*/{
                                    stencil::AccessOp::getOffsetAttrName()});
  printer << " : (";
  printer.printType(temp.getType());
  printer << ") -> ";
  printer.printType(accessOp.getResult().getType());
}

static LogicalResult verify(stencil::AccessOp accessOp) {
  stencil::TempType tempType =
      accessOp.temp().getType().cast<stencil::TempType>();
  Type elementType = tempType.getElementType();
  Type resultType = accessOp.getResult().getType();

  if (resultType != elementType)
    return accessOp.emitOpError("inconsistent result type '")
           << resultType << "' and element type '" << elementType << "'";

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.load
//===----------------------------------------------------------------------===//

void stencil::LoadOp::build(OpBuilder &builder, OperationState &state,
                            Value field) {
  Type elementType =
      field.getType().cast<stencil::FieldType>().getElementType();
  ArrayRef<int> dimensions =
      field.getType().cast<stencil::FieldType>().getDimensions();

  state.addOperands(field);
  state.addTypes(
      stencil::TempType::get(builder.getContext(), elementType, dimensions));
}

static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &state) {
  FunctionType funcType;
  SmallVector<OpAsmParser::OperandType, 1> operands;
  ArrayAttr lbAttr, ubAttr;

  if (parser.parseOperandList(operands))
    return failure();
  if (!parser.parseOptionalLParen()) {
    // Parse the optional bounds
    if (parser.parseAttribute(lbAttr, stencil::LoadOp::getLBAttrName(),
                              state.attributes) ||
        parser.parseColon() ||
        parser.parseAttribute(ubAttr, stencil::LoadOp::getUBAttrName(),
                              state.attributes) ||
        parser.parseRParen())
      return failure();

    // Make sure bounds have the right number of dimensions
    if (lbAttr.size() != stencil::kNumOfDimensions ||
        ubAttr.size() != stencil::kNumOfDimensions) {
      parser.emitError(
          parser.getCurrentLocation(),
          "expected bounds to have a component for every dimension");
      return failure();
    }
  }

  if (parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType<FunctionType>(funcType) ||
      parser.resolveOperands(operands, funcType.getInputs(),
                             parser.getCurrentLocation(), state.operands) ||
      parser.addTypesToList(funcType.getResults(), state.types))
    return failure();

  return success();
}

static void print(stencil::LoadOp loadOp, OpAsmPrinter &printer) {
  Value field = loadOp.field();
  Type fieldType = field.getType();
  Type tempType = loadOp.res().getType();

  printer << stencil::LoadOp::getOperationName() << ' ' << field;
  if (loadOp.lb().hasValue() && loadOp.ub().hasValue()) {
    printer << " (";
    printer.printAttribute(loadOp.lb().getValue());
    printer << ":";
    printer.printAttribute(loadOp.ub().getValue());
    printer << ")";
  }
  printer.printOptionalAttrDict(
      loadOp.getAttrs(), /*elidedAttrs=*/{stencil::LoadOp::getLBAttrName(),
                                          stencil::LoadOp::getUBAttrName()});
  printer << " : (";
  printer.printType(fieldType);
  printer << ") -> ";
  printer.printType(tempType);
}

static LogicalResult verify(stencil::LoadOp loadOp) {
  // Check the field and temp types match
  stencil::FieldType fieldType =
      loadOp.field().getType().cast<stencil::FieldType>();
  stencil::TempType tempType = loadOp.res().getType().cast<stencil::TempType>();

  Type fieldElementType = fieldType.getElementType();
  Type tempElementType = tempType.getElementType();
  if (fieldElementType != tempElementType)
    return loadOp.emitOpError("inconsistent field element type '")
           << fieldElementType << "' and temp element type '" << tempElementType
           << "'";

  auto fieldDimensions = fieldType.getDimensions();
  auto tempDimensions = tempType.getDimensions();
  if (fieldDimensions != tempDimensions)
    return loadOp.emitOpError("storage dimensions are inconsistent");

  // Check if field assert exists
  int asserts = 0;
  for (OpOperand &use : loadOp.field().getUses()) {
    if (auto assertOp = dyn_cast<stencil::AssertOp>(use.getOwner()))
      asserts++;
  }
  if (asserts != 1)
    return loadOp.emitOpError("assert for input field missing");

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.store
//===----------------------------------------------------------------------===//

void stencil::StoreOp::build(OpBuilder &builder, OperationState &state,
                             Value temp, Value field, ArrayRef<int64_t> lb,
                             ArrayRef<int64_t> ub) {
  // Make sure that the offset has the right size
  assert(lb.size() == stencil::kNumOfDimensions &&
         ub.size() == stencil::kNumOfDimensions &&
         "expected bounds to have an element for every dimension");

  // Add an SSA arguments
  state.addOperands({temp, field});
  // Add the bounds attributes
  state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
  state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub));
}

static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType temp, field;
  ArrayAttr lbAttr, ubAttr;
  Type fieldType, tempType;
  // Parse the store op
  if (parser.parseOperand(temp) || parser.parseKeyword("to") ||
      parser.parseOperand(field) || parser.parseLParen() ||
      parser.parseAttribute(lbAttr, stencil::StoreOp::getLBAttrName(),
                            state.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(ubAttr, stencil::StoreOp::getUBAttrName(),
                            state.attributes) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColon() || parser.parseType(tempType) ||
      parser.parseKeyword("to") || parser.parseType(fieldType))
    return failure();

  // Make sure bounds have the right number of dimensions
  if (lbAttr.size() != stencil::kNumOfDimensions ||
      ubAttr.size() != stencil::kNumOfDimensions) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected bounds to have a component for every dimension");
    return failure();
  }

  if (parser.resolveOperand(temp, tempType, state.operands) ||
      parser.resolveOperand(field, fieldType, state.operands))
    return failure();

  return success();
}

static void print(stencil::StoreOp storeOp, OpAsmPrinter &printer) {
  Value field = storeOp.field();
  Value temp = storeOp.temp();
  ArrayAttr lb = storeOp.lb();
  ArrayAttr ub = storeOp.ub();

  printer << stencil::StoreOp::getOperationName() << " " << temp;
  printer << " to " << field << " (";
  printer.printAttribute(lb);
  printer << ":";
  printer.printAttribute(ub);
  printer << ")";
  printer.printOptionalAttrDict(
      storeOp.getAttrs(), /*elidedAttrs=*/{stencil::StoreOp::getLBAttrName(),
                                           stencil::StoreOp::getUBAttrName()});
  printer << " : ";
  printer.printType(temp.getType());
  printer << " to ";
  printer.printType(field.getType());
}

static LogicalResult verify(stencil::StoreOp storeOp) {
  // Check the field and temp types match
  stencil::FieldType fieldType = storeOp.getFieldType();
  stencil::TempType tempType = storeOp.getTempType();

  Type fieldElementType = fieldType.getElementType();
  Type tempElementType = tempType.getElementType();
  if (fieldElementType != tempElementType)
    return storeOp.emitOpError("inconsistent field element type '")
           << fieldElementType << "' and temp element type '" << tempElementType
           << "'";

  auto fieldDimensions = fieldType.getDimensions();
  auto tempDimensions = tempType.getDimensions();
  if (fieldDimensions != tempDimensions)
    return storeOp.emitOpError("storage dimensions are inconsistent");

  // Check temp computed by apply
  if (!dyn_cast<stencil::ApplyOp>(storeOp.temp().getDefiningOp()))
    return storeOp.emitError("output temp not result of an apply");

  // Check if field assert exists
  int asserts = 0;
  for (OpOperand &use : storeOp.field().getUses()) {
    if (auto assertOp = dyn_cast<stencil::AssertOp>(use.getOwner()))
      asserts++;
  }
  if (asserts != 1)
    return storeOp.emitOpError("assert for output field missing");

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.apply
//===----------------------------------------------------------------------===//

void stencil::ApplyOp::build(OpBuilder &builder, OperationState &result,
                             ValueRange operands, ValueRange results) {
  result.addOperands(operands);

  // Create an empty body
  Region *body = result.addRegion();
  // ensureTerminator(*body, *builder, result.location);
  // for (auto operand : operands) {
  //   body->front().addArgument(operand.getType());
  // }

  // Add result types
  SmallVector<Type, 3> resultTypes;
  resultTypes.reserve(results.size());
  for (auto result : results) {
    resultTypes.push_back(result.getType());
  }
  result.addTypes(resultTypes);
}

static ParseResult parseApplyOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 8> operands;
  SmallVector<OpAsmParser::OperandType, 8> arguments;
  ArrayAttr lbAttr, ubAttr;

  // Parse region arguments and the assigned data operands
  llvm::SMLoc loc = parser.getCurrentLocation();
  do {
    OpAsmParser::OperandType currentArgument;
    OpAsmParser::OperandType currentOperand;
    if (parser.parseRegionArgument(currentArgument) || parser.parseEqual() ||
        parser.parseOperand(currentOperand))
      return failure();
    arguments.push_back(currentArgument);
    operands.push_back(currentOperand);
  } while (!parser.parseOptionalComma());

  // Parse optional attributes and the operand types
  SmallVector<Type, 8> operandTypes;
  if (parser.parseColonTypeList(operandTypes))
    return failure();

  if (operands.size() != operandTypes.size()) {
    parser.emitError(parser.getCurrentLocation(), "expected ")
        << operands.size() << " operand types";
    return failure();
  }

  // Parse the body region.
  SmallVector<Type, 8> resultTypes;
  Region *body = state.addRegion();
  if (parser.parseRegion(*body, arguments, operandTypes))
    return failure();

  // Parse the optional bounds
  if (!parser.parseOptionalKeyword("to")) {
    // Parse the optional bounds
    if (parser.parseLParen() ||
        parser.parseAttribute(lbAttr, stencil::LoadOp::getLBAttrName(),
                              state.attributes) ||
        parser.parseColon() ||
        parser.parseAttribute(ubAttr, stencil::LoadOp::getUBAttrName(),
                              state.attributes) ||
        parser.parseRParen())
      return failure();

    // Make sure bounds have the right number of dimensions
    if (lbAttr.size() != stencil::kNumOfDimensions ||
        ubAttr.size() != stencil::kNumOfDimensions) {
      parser.emitError(
          parser.getCurrentLocation(),
          "expected bounds to have a component for every dimension");
      return failure();
    }
  }

  // Parse the return types and resolve all operands
  if (parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonTypeList(resultTypes) ||
      parser.resolveOperands(operands, operandTypes, loc, state.operands) ||
      parser.addTypesToList(resultTypes, state.types))
    return failure();

  return success();
}

static void print(stencil::ApplyOp applyOp, OpAsmPrinter &printer) {
  printer << stencil::ApplyOp::getOperationName() << ' ';

  // Print the region arguments
  ValueRange operands = applyOp.getOperands();
  if (!applyOp.region().empty() && !operands.empty()) {
    Block *body = applyOp.getBody();
    llvm::interleaveComma(llvm::seq<int>(0, operands.size()), printer, [&](int i) {
      printer << body->getArgument(i) << " = " << operands[i];
    });
  }

  // Print the operand types
  printer << " : ";
  llvm::interleaveComma(applyOp.getOperandTypes(), printer);

  // Print region, bounds, and return type
  printer.printRegion(applyOp.region(),
                      /*printEntryBlockArgs=*/false);
  if (applyOp.lb().hasValue() && applyOp.ub().hasValue()) {
    printer << " to (";
    printer.printAttribute(applyOp.lb().getValue());
    printer << ":";
    printer.printAttribute(applyOp.ub().getValue());
    printer << ")";
  }
  printer.printOptionalAttrDict(
      applyOp.getAttrs(), /*elidedAttrs=*/{stencil::ApplyOp::getLBAttrName(),
                                           stencil::ApplyOp::getUBAttrName()});
  printer << " : ";
  llvm::interleaveComma(applyOp.res().getTypes(), printer);
}

static LogicalResult verify(stencil::ApplyOp applyOp) {
  // Check the body takes at least one argument
  auto *body = applyOp.getBody();
  if (body->getNumArguments() == 0)
    return applyOp.emitOpError("expected body to have at least one argument");

  // TODO check the body contains only valid operations

  // Check the number of operands and arguments match
  if (body->getNumArguments() != applyOp.operands().size())
    return applyOp.emitOpError(
        "expected operation and body to have same number of arguments");

  // Check the operands match the block argument types
  for (unsigned i = 0, e = applyOp.operands().size(); i != e; ++i) {
    if (applyOp.getBody()->getArgument(i).getType() !=
        applyOp.operands()[i].getType())
      return applyOp.emitOpError(
          "expected operation and body arguments to have the same type");
  }
  return success();
}

namespace {
/// This is a pattern to remove duplicate results
struct ApplyOpResCleaner : public OpRewritePattern<stencil::ApplyOp> {
  using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Get the terminator and compute the result list
    auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
    unsigned unrollFactor = returnOp.getUnrollFactor();
    unsigned numResults = returnOp.getNumOperands() / unrollFactor;
    assert(returnOp.getNumOperands() % unrollFactor == 0 &&
           "expected number of operands to be a multiple of the unroll factor");
    SmallVector<Value, 10> oldOperands = returnOp.getOperands();
    SmallVector<Value, 10> newOperands;
    SmallVector<Value, 10> newResults;
    for (unsigned i = 0, e = numResults; i != e; ++i) {
      // Get the operands for every result
      auto operands =
          returnOp.getOperands().slice(i * unrollFactor, unrollFactor);
      if (!llvm::all_of(operands, [&](Value value) {
            return llvm::is_contained(newOperands, value);
          })) {
        newOperands.insert(newOperands.end(), operands.begin(), operands.end());
        newResults.push_back(applyOp.getResult(i));
      }
    }

    // Remove duplicates if needed
    if (newOperands.size() < returnOp.getNumOperands()) {
      // Replace the return op
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<stencil::ReturnOp>(returnOp.getLoc(), newOperands,
                                         returnOp.unroll().getValueOr(nullptr));
      rewriter.eraseOp(returnOp);

      // Clone the apply op
      rewriter.setInsertionPoint(applyOp);
      auto newOp = rewriter.create<stencil::ApplyOp>(
          applyOp.getLoc(), applyOp.getOperands(), newResults);
      rewriter.cloneRegionBefore(applyOp.region(), newOp.region(),
                                 newOp.region().begin());

      // Replace all uses of the applyOp results
      SmallVector<Value, 10> repResults;
      for (size_t i = 0, e = applyOp.getResults().size(); i != e; ++i) {
        auto it = llvm::find(newOperands, oldOperands[i]);
        repResults.push_back(
            newOp.getResult(std::distance(newOperands.begin(), it)));
      }
      rewriter.replaceOp(applyOp, repResults);
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
    // Compute operand list and init the argument matcher
    BlockAndValueMapping mapper;
    SmallVector<Value, 10> newOperands;
    for (unsigned i = 0, e = applyOp.getNumOperands(); i != e; ++i) {
      if (llvm::is_contained(newOperands, applyOp.getOperand(i))) {
        mapper.map(applyOp.getBody()->getArgument(i),
                   applyOp.getBody()->getArgument(i));
      } else {
        newOperands.push_back(applyOp.getOperand(i));
      }
    }

    if (newOperands.size() < applyOp.getNumOperands()) {
      // Replace all uses of duplicates
      for (unsigned i = 0, e = applyOp.getNumOperands(); i != e; ++i) {
        if (mapper.contains(applyOp.getBody()->getArgument(i))) {
          auto it = llvm::find(applyOp.getOperands(), applyOp.getOperand(i));
          size_t index = std::distance(applyOp.getOperands().begin(), it);
          assert(index < i && "expected lower replacement index");
          applyOp.getBody()->getArgument(i).replaceAllUsesWith(
              applyOp.getBody()->getArgument(index));
        }
      }

      // Clone the apply op
      auto loc = applyOp.getLoc();
      auto newOp = rewriter.create<stencil::ApplyOp>(loc, newOperands,
                                                     applyOp.getResults());
      rewriter.cloneRegionBefore(applyOp.region(), newOp.region(),
                                 newOp.region().begin(), mapper);

      // Replace all uses of the applyOp results
      rewriter.replaceOp(applyOp, newOp.getResults());
      return success();
    }
    return failure();
  }
};
} // end anonymous namespace

void stencil::ApplyOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ApplyOpArgCleaner, ApplyOpResCleaner>(context);
}

//===----------------------------------------------------------------------===//
// stencil.return
//===----------------------------------------------------------------------===//

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  SmallVector<Type, 2> operandTypes;
  llvm::SMLoc loc = parser.getCurrentLocation();
  ArrayAttr unrollAttr;

  // Parse optional unroll attr
  if (!parser.parseOptionalKeyword("unroll")) {
    if (parser.parseAttribute(unrollAttr,
                              stencil::ReturnOp::getUnrollAttrName(),
                              result.attributes))
      return failure();

    // Make sure unroll parameters have right number of dimensions
    if (unrollAttr.size() != stencil::kNumOfDimensions) {
      parser.emitError(
          parser.getCurrentLocation(),
          "expected unroll attribute to have a component for every dimension");
      return failure();
    }
  }

  return failure(
      parser.parseOperandList(operands) ||
      parser.parseColonTypeList(operandTypes) ||
      parser.resolveOperands(operands, operandTypes, loc, result.operands));
}

static void print(stencil::ReturnOp returnOp, OpAsmPrinter &printer) {
  printer << stencil::ReturnOp::getOperationName() << ' ';
  if (returnOp.unroll().hasValue()) {
    printer << "unroll ";
    printer.printAttribute(returnOp.unroll().getValue());
    printer << " ";
  }
  printer << returnOp.getOperands() << " : " << returnOp.getOperandTypes();
}

static LogicalResult verify(stencil::ReturnOp returnOp) {
  auto applyOp = cast<stencil::ApplyOp>(returnOp.getParentOp());
  unsigned unrollFactor = returnOp.getUnrollFactor();

  // The operand number and types times the unroll factor must match the apply
  // signature
  auto results = applyOp.res();
  if (returnOp.getNumOperands() != unrollFactor * results.size())
    return returnOp.emitOpError("has ")
           << returnOp.getNumOperands()
           << " operands, but enclosing function returns "
           << unrollFactor * results.size();

  // The return types must match the element types of the returned temps
  for (unsigned i = 0, e = results.size(); i != e; ++i) {
    for (unsigned j = 0; j < unrollFactor; j++)
      if (returnOp.getOperand(i * unrollFactor + j).getType() !=
          applyOp.getResultTempType(i).getElementType())
        return returnOp.emitError()
               << "type of return operand " << i * unrollFactor + j << " ("
               << returnOp.getOperand(i * unrollFactor + j).getType()
               << ") doesn't match function result type ("
               << applyOp.getResultTempType(i).getElementType() << ")";
  }

  return success();
}

namespace mlir {
namespace stencil {
#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.cpp.inc"
} // namespace stencil
} // namespace mlir
