#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>

using namespace mlir;

//===----------------------------------------------------------------------===//
// stencil.assert
//===----------------------------------------------------------------------===//

void stencil::AssertOp::build(Builder *builder, OperationState &state,
                              Value field, ArrayRef<int64_t> lb,
                              ArrayRef<int64_t> ub) {
  // Make sure that the offset has the right size
  assert(lb.size() == 3 && ub.size() == 3 && "expected bounds with 3 elements");

  // Add an SSA arguments
  state.addOperands({field});
  // Add the bounds attributes
  state.addAttribute(getLBAttrName(), builder->getI64ArrayAttr(lb));
  state.addAttribute(getUBAttrName(), builder->getI64ArrayAttr(ub));
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
  if (lbAttr.size() != 3 || ubAttr.size() != 3) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected bounds to have three components");
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
  printer << " " << *field << " (";
  printer.printAttribute(lb);
  printer << ":";
  printer.printAttribute(ub);
  printer << ")";
  printer.printOptionalAttrDict(assertOp.getAttrs(), /*elidedAttrs=*/{
                                    stencil::AssertOp::getLBAttrName(),
                                    stencil::AssertOp::getUBAttrName()});
  printer << " : ";
  printer.printType(assertOp.field()->getType());
}

static LogicalResult verify(stencil::AssertOp assertOp) {
  // Check if all uses are loads or stores
  int stores = 0;
  int loads = 0;
  int asserts = 0;
  for (OpOperand &use : assertOp.field()->getUses()) {
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

  // Check if the field is large enough
  auto verifyBounds = [&](const SmallVector<int64_t, 3> &lb,
                          const SmallVector<int64_t, 3> &ub) {
    if (llvm::any_of(llvm::zip(lb, assertOp.getLB()),
                     [](std::tuple<int64_t, int64_t> x) {
                       return std::get<0>(x) < std::get<1>(x);
                     }) ||
        llvm::any_of(llvm::zip(ub, assertOp.getUB()),
                     [](std::tuple<int64_t, int64_t> x) {
                       return std::get<0>(x) > std::get<1>(x);
                     }))
      return false;
    return true;
  };
  for (OpOperand &use : assertOp.field()->getUses()) {
    if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner()))
      if (!verifyBounds(storeOp.getLB(), storeOp.getUB())) {
        return assertOp.emitOpError("field bounds not large enough");
      }
    if (auto loadOp = dyn_cast<stencil::LoadOp>(use.getOwner()))
      if (loadOp.lb().hasValue() && loadOp.ub().hasValue())
        if (!verifyBounds(loadOp.getLB(), loadOp.getUB())) {
          return assertOp.emitOpError("field bounds not large enough");
        }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.access
//===----------------------------------------------------------------------===//

void stencil::AccessOp::build(Builder *builder, OperationState &state,
                              Value view, ArrayRef<int64_t> offset) {
  // Make sure that the offset has the right size
  assert(offset.size() == 3 && "expected offset with 3 elements");

  // Extract the element type of the view.
  Type elementType = view->getType().cast<stencil::ViewType>().getElementType();

  // Add an SSA argument
  state.addOperands(view);
  // Add the offset attribute
  state.addAttribute(getOffsetAttrName(), builder->getI64ArrayAttr(offset));
  // Set the return type
  state.addTypes(elementType);
}

static ParseResult parseAccessOp(OpAsmParser &parser, OperationState &state) {
  FunctionType funcType;
  ArrayAttr offset;
  SmallVector<OpAsmParser::OperandType, 1> operands;

  // Parse the view
  if (parser.parseOperandList(operands) ||
      parser.parseAttribute(offset, stencil::AccessOp::getOffsetAttrName(),
                            state.attributes))
    return failure();
  // Make sure it has the right number of dimensions
  if (offset.size() != 3) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected offset to have three components");
    return failure();
  }

  // Parse optional attributes as well as the view type
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
  // Use the TableGen'd accessors to operands
  Value view = accessOp.view();
  Attribute offset = accessOp.offset();

  printer << stencil::AccessOp::getOperationName() << ' ' << *view;
  printer.printAttribute(offset);
  printer.printOptionalAttrDict(accessOp.getAttrs(), /*elidedAttrs=*/{
                                    stencil::AccessOp::getOffsetAttrName()});
  printer << " : (";
  printer.printType(view->getType());
  printer << ") -> ";
  printer.printType(accessOp.getResult()->getType());
}

static LogicalResult verify(stencil::AccessOp accessOp) {
  stencil::ViewType viewType =
      accessOp.view()->getType().cast<stencil::ViewType>();
  Type elementType = viewType.getElementType();
  Type resultType = accessOp.getResult()->getType();

  if (resultType != elementType)
    return accessOp.emitOpError("inconsistent result type '")
           << resultType << "' and element type '" << elementType << "'";

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.load
//===----------------------------------------------------------------------===//

void stencil::LoadOp::build(Builder *builder, OperationState &state,
                            Value field) {
  Type elementType =
      field->getType().cast<stencil::FieldType>().getElementType();
  ArrayRef<int> dimensions =
      field->getType().cast<stencil::FieldType>().getDimensions();

  state.addOperands(field);
  state.addTypes(
      stencil::ViewType::get(builder->getContext(), elementType, dimensions));
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
    if (lbAttr.size() != 3 || ubAttr.size() != 3) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected bounds to have three components");
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
  Type fieldType = field->getType();
  Type viewType = loadOp.res()->getType();

  printer << stencil::LoadOp::getOperationName() << ' ' << *field;
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
  printer.printType(viewType);
}

static LogicalResult verify(stencil::LoadOp loadOp) {
  // Check the field and view types match
  stencil::FieldType fieldType =
      loadOp.field()->getType().cast<stencil::FieldType>();
  stencil::ViewType viewType =
      loadOp.res()->getType().cast<stencil::ViewType>();

  Type fieldElementType = fieldType.getElementType();
  Type viewElementType = viewType.getElementType();
  if (fieldElementType != viewElementType)
    return loadOp.emitOpError("inconsistent field element type '")
           << fieldElementType << "' and view element type '" << viewElementType
           << "'";

  auto fieldDimensions = fieldType.getDimensions();
  auto viewDimensions = viewType.getDimensions();
  if (fieldDimensions != viewDimensions)
    return loadOp.emitOpError("storage dimensions are inconsistent");

  // Check if field assert exists
  int asserts = 0;
  for (OpOperand &use : loadOp.field()->getUses()) {
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

void stencil::StoreOp::build(Builder *builder, OperationState &state,
                             Value view, Value field, ArrayRef<int64_t> lb,
                             ArrayRef<int64_t> ub) {
  // Make sure that the offset has the right size
  assert(lb.size() == 3 && ub.size() == 3 && "expected bounds with 3 elements");

  // Add an SSA arguments
  state.addOperands({view, field});
  // Add the bounds attributes
  state.addAttribute(getLBAttrName(), builder->getI64ArrayAttr(lb));
  state.addAttribute(getUBAttrName(), builder->getI64ArrayAttr(ub));
}

static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType view, field;
  ArrayAttr lbAttr, ubAttr;
  Type fieldType, viewType;
  // Parse the store op
  if (parser.parseOperand(view) || parser.parseKeyword("to") ||
      parser.parseOperand(field) || parser.parseLParen() ||
      parser.parseAttribute(lbAttr, stencil::StoreOp::getLBAttrName(),
                            state.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(ubAttr, stencil::StoreOp::getUBAttrName(),
                            state.attributes) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColon() || parser.parseType(viewType) ||
      parser.parseKeyword("to") || parser.parseType(fieldType))
    return failure();

  // Make sure bounds have the right number of dimensions
  if (lbAttr.size() != 3 || ubAttr.size() != 3) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected bounds to have three components");
    return failure();
  }

  if (parser.resolveOperand(view, viewType, state.operands) ||
      parser.resolveOperand(field, fieldType, state.operands))
    return failure();

  return success();
}

static void print(stencil::StoreOp storeOp, OpAsmPrinter &printer) {
  Value field = storeOp.field();
  Value view = storeOp.view();
  ArrayAttr lb = storeOp.lb();
  ArrayAttr ub = storeOp.ub();

  printer << stencil::StoreOp::getOperationName() << " " << *view;
  printer << " to " << *field << " (";
  printer.printAttribute(lb);
  printer << ":";
  printer.printAttribute(ub);
  printer << ")";
  printer.printOptionalAttrDict(
      storeOp.getAttrs(), /*elidedAttrs=*/{stencil::StoreOp::getLBAttrName(),
                                           stencil::StoreOp::getUBAttrName()});
  printer << " : ";
  printer.printType(view->getType());
  printer << " to ";
  printer.printType(field->getType());
}

static LogicalResult verify(stencil::StoreOp storeOp) {
  // Check the field and view types match
  stencil::FieldType fieldType = storeOp.getFieldType();
  stencil::ViewType viewType = storeOp.getViewType();

  Type fieldElementType = fieldType.getElementType();
  Type viewElementType = viewType.getElementType();
  if (fieldElementType != viewElementType)
    return storeOp.emitOpError("inconsistent field element type '")
           << fieldElementType << "' and view element type '" << viewElementType
           << "'";

  auto fieldDimensions = fieldType.getDimensions();
  auto viewDimensions = viewType.getDimensions();
  if (fieldDimensions != viewDimensions)
    return storeOp.emitOpError("storage dimensions are inconsistent");

  // Check view computed by apply
  if (!dyn_cast<stencil::ApplyOp>(storeOp.view()->getDefiningOp()))
    return storeOp.emitError("output view not result of an apply");

  // Check if field assert exists
  int asserts = 0;
  for (OpOperand &use : storeOp.field()->getUses()) {
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

void stencil::ApplyOp::build(Builder *builder, OperationState &result,
                             ValueRange operands, ValueRange results) {
  result.addOperands(operands);

  // Create an empty body
  Region *body = result.addRegion();
  ensureTerminator(*body, *builder, result.location);
  for(auto operand : operands) {
    body->front().addArgument(operand->getType());
  }

  // Add result types
  SmallVector<Type, 3> resultTypes;
  resultTypes.reserve(results.size());
  for(auto result : results) {
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
    if (lbAttr.size() != 3 || ubAttr.size() != 3) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected bounds to have three components");
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
    interleaveComma(llvm::seq<int>(0, operands.size()), printer, [&](int i) {
      printer << *body->getArgument(i) << " = " << *operands[i];
    });
  }

  // Print the operand types
  printer << " : ";
  interleaveComma(applyOp.getOperandTypes(), printer);

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
  interleaveComma(applyOp.res().getTypes(), printer);
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
    if (applyOp.getBody()->getArgument(i)->getType() !=
        applyOp.operands()[i]->getType())
      return applyOp.emitOpError(
          "expected operation and body arguments to have the same type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stencil.call
//===----------------------------------------------------------------------===//

void stencil::CallOp::build(Builder *builder, OperationState &result,
                            FuncOp callee, stencil::ViewType viewType,
                            ArrayRef<int64_t> offset,
                            ValueRange operands) {
  assert(offset.size() == 3 && "expected offset with 3 elements");
  assert(
      callee.getAttr(stencil::StencilDialect::getStencilFunctionAttrName()) &&
      "only stencil functions can be used in an apply operation");
  assert(callee.getType().getNumResults() == 1 &&
         "expected stencil function to return only one result");
  assert(callee.getType().getResult(0) == viewType.getElementType() &&
         "incompatible stencil function return type "
         "and view type");
ValueRange test;
  result.addOperands(operands);
  result.addAttribute(getCalleeAttrName(), builder->getSymbolRefAttr(callee));
  result.addAttribute(getOffsetAttrName(), builder->getI64ArrayAttr(offset));
  result.addTypes(viewType);
}

FunctionType stencil::CallOp::getCalleeType() {
  SmallVector<Type, 1> resultTypes({getResult()->getType()});
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &state) {
  SymbolRefAttr calleeAttr;
  FunctionType funcType;
  SmallVector<OpAsmParser::OperandType, 3> operands;
  ArrayAttr offset;
  auto calleeLoc = parser.getNameLoc();
  if (parser.parseAttribute(calleeAttr, stencil::CallOp::getCalleeAttrName(),
                            state.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseAttribute(offset, stencil::CallOp::getOffsetAttrName(),
                            state.attributes) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType(funcType) ||
      parser.addTypesToList(funcType.getResults(), state.types) ||
      parser.resolveOperands(operands, funcType.getInputs(), calleeLoc,
                             state.operands))
    return failure();

  return success();
}

static void print(stencil::CallOp callOp, OpAsmPrinter &printer) {
  printer << stencil::CallOp::getOperationName() << ' '
          << callOp.getAttr(stencil::CallOp::getCalleeAttrName()) << '(';
  printer.printOperands(callOp.getOperands());
  printer << ')' << callOp.getAttr(stencil::CallOp::getOffsetAttrName());
  printer.printOptionalAttrDict(callOp.getAttrs(),
                                {stencil::CallOp::getCalleeAttrName(),
                                 stencil::CallOp::getOffsetAttrName()});
  printer << " : ";

  FunctionType calleeType = callOp.getCalleeType();
  printer.printType(calleeType);
}

static LogicalResult verify(stencil::CallOp callOp) {
  // Check that the callee attribute was specified.
  auto funAttr =
      callOp.getAttrOfType<SymbolRefAttr>(stencil::CallOp::getCalleeAttrName());
  if (!funAttr)
    return callOp.emitOpError("requires a '")
           << stencil::CallOp::getCalleeAttrName()
           << "' symbol reference attribute";
  auto fun = callOp.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      funAttr.getLeafReference());
  if (!fun)
    return callOp.emitOpError() << "'" << funAttr.getLeafReference()
                                << "' does not reference a valid function";
  if (!fun.getAttr(stencil::StencilDialect::getStencilFunctionAttrName()))
    return callOp.emitOpError() << "'" << funAttr.getLeafReference()
                                << "' does not reference a stencil function";

  // Verify that the operand and result types match the callee.
  auto funType = fun.getType();
  if (funType.getNumInputs() != callOp.getNumOperands())
    return callOp.emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = funType.getNumInputs(); i != e; ++i)
    if (callOp.getOperand(i)->getType() != funType.getInput(i))
      return callOp.emitOpError("operand type mismatch");

  if (funType.getNumResults() != 1)
    return callOp.emitOpError("incorrect number of results for callee");

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.return
//===----------------------------------------------------------------------===//

void stencil::ReturnOp::build(Builder *builder, OperationState &result) {
  result.addOperands({});
  result.addTypes({});
}

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  SmallVector<Type, 2> operandTypes;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(
      parser.parseOperandList(operands) ||
      parser.parseColonTypeList(operandTypes) ||
      parser.resolveOperands(operands, operandTypes, loc, result.operands));
}

static void print(stencil::ReturnOp returnOp, OpAsmPrinter &printer) {
  printer << stencil::ReturnOp::getOperationName() << ' ';
  printer << returnOp.getOperands() << " : " << returnOp.getOperandTypes();
}

static LogicalResult verify(stencil::ReturnOp returnOp) {
  auto applyOp = cast<stencil::ApplyOp>(returnOp.getParentOp());

  // The operand number and types must match the apply signature
  const auto &results = applyOp.res();
  if (returnOp.getNumOperands() != results.size())
    return returnOp.emitOpError("has ")
           << returnOp.getNumOperands()
           << " operands, but enclosing function returns " << results.size();

  // The return types must match the element types of the returned views
  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (returnOp.getOperand(i)->getType() !=
        applyOp.getResultViewType(i).getElementType())
      return returnOp.emitError()
             << "type of return operand " << i << " ("
             << returnOp.getOperand(i)->getType()
             << ") doesn't match function result type ("
             << applyOp.getResultViewType(i).getElementType() << ")";

  return success();
}

namespace mlir {
namespace stencil {
#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.cpp.inc"
} // namespace stencil
} // namespace mlir
