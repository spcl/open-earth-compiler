#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// stencil.access
//===----------------------------------------------------------------------===//

void stencil::AccessOp::build(Builder *builder, OperationState &state,
                              Value *view, ArrayRef<int64_t> offset) {
  // Make sure that the offset has the right size
  assert(offset.size() == 3 && "expected offset with 3 elements");

  // Extract the element type of the view.
  // The `cast` operation will fail and throw an error if the type of the
  // view is not `stencil::ViewType`.
  Type elementType = view->getType().cast<stencil::ViewType>().getElementType();

  // Add an SSA argument
  state.addOperands(view);
  // Add the offset attribute
  state.addAttribute(getOffsetAttrName(), builder->getI64ArrayAttr(offset));
  // Set the return type
  state.addTypes(elementType);
}

static ParseResult parseAccessOp(OpAsmParser &parser, OperationState &state) {
  // %0 = stencil.access %view[-1, 0, 0] : !stencil.view<?x?x?xf64>

  stencil::ViewType viewType;
  Type elementType;
  ArrayAttr offset;
  // SSA values have the special type `OpAsmParser::OperandType` until they are
  // resolved
  OpAsmParser::OperandType view;

  // Step 1: Parse the `%view` operand
  if (parser.parseOperand(view))
    return failure();
  // Step 2: Parse the `offset` attribute
  if (parser.parseAttribute(offset, stencil::AccessOp::getOffsetAttrName(),
                            state.attributes))
    return failure();
  // Step 2.5: Make sure it has the right number of dimensions
  if (offset.size() != 3) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected offset to have three components");
    return failure();
  }

  // Step 3: Parse optional attributes as well as the view type
  if (parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType<stencil::ViewType>(viewType))
    return failure();
  // Step 4: Make sure the `%view` operand is of the right type
  if (parser.resolveOperand(view, viewType, state.operands))
    return failure();

  // Step 5: Extract the element type from the view type
  elementType = viewType.getElementType();
  // Step 6: Add the return value
  if (parser.addTypeToList(elementType, state.types))
    return failure();

  return success();
}

static void print(stencil::AccessOp accessOp, OpAsmPrinter &printer) {
  // Use the TableGen'd accessors to operands
  Value *view = accessOp.view();
  Attribute offset = accessOp.offset();

  printer << stencil::AccessOp::getOperationName() << ' ' << *view;
  printer.printAttribute(offset);
  printer.printOptionalAttrDict(accessOp.getAttrs(), /*elidedAttrs=*/{
                                    stencil::AccessOp::getOffsetAttrName()});
  printer << " : ";
  printer.printType(view->getType());
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
                            Value *field, ArrayRef<int64_t> viewShape) {
  assert(viewShape.size() == 3 && "view shape should have three components");

  Type elementType =
      field->getType().cast<stencil::FieldType>().getElementType();

  state.addOperands(field);
  state.addTypes(
      stencil::ViewType::get(builder->getContext(), elementType, viewShape));
}

static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &state) {
  FunctionType funcType;
  OpAsmParser::OperandType field;

  if (parser.parseOperand(field) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType<FunctionType>(funcType))
    return failure();

  if (parser.resolveOperands(field, funcType.getInputs(),
                             parser.getCurrentLocation(), state.operands) ||
      parser.addTypesToList(funcType.getResults(), state.types))
    return failure();

  return success();
}

static void print(stencil::LoadOp loadOp, OpAsmPrinter &printer) {
  Value *field = loadOp.field();

  Type fieldType = field->getType();
  Type viewType = loadOp.res()->getType();

  printer << stencil::LoadOp::getOperationName() << ' ' << *field;
  printer << " : (";
  printer.printType(fieldType);
  printer << ") -> ";
  printer.printType(viewType);
}

static LogicalResult verify(stencil::LoadOp loadOp) {
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

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.store
//===----------------------------------------------------------------------===//

static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType view, field;
  Type viewType, fieldType;

  if (parser.parseOperand(view) || parser.parseKeyword("to") ||
      parser.parseOperand(field) ||
      parser.parseOptionalAttrDict(state.attributes) || parser.parseColon() ||
      parser.parseType(viewType) || parser.parseComma() ||
      parser.parseType(fieldType))
    return failure();

  if (parser.resolveOperand(view, viewType, state.operands) ||
      parser.resolveOperand(field, fieldType, state.operands))
    return failure();

  return success();
}

static void print(stencil::StoreOp storeOp, OpAsmPrinter &printer) {
  Value *field = storeOp.field();
  Value *view = storeOp.view();
  printer << stencil::StoreOp::getOperationName() << ' ' << *view << " to "
          << *field << " : ";
  printer.printType(view->getType());
  printer << ", ";
  printer.printType(field->getType());
}

static LogicalResult verify(stencil::StoreOp storeOp) {
  stencil::FieldType fieldType = storeOp.getFieldType();
  stencil::ViewType viewType = storeOp.getViewType();

  Type fieldElementType = fieldType.getElementType();
  Type viewElementType = viewType.getElementType();
  if (fieldElementType != viewElementType)
    return storeOp.emitOpError("inconsistent field element type '")
           << fieldElementType << "' and view element type '" << viewElementType
           << "'";

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.apply
//===----------------------------------------------------------------------===//

void stencil::ApplyOp::build(Builder *builder, OperationState &result,
                             FuncOp callee, stencil::ViewType viewType,
                             ArrayRef<Value *> operands) {
  if (!callee.getAttr(stencil::StencilDialect::getStencilFunctionAttrName()))
    callee.emitError(
        "only stencil functions can be used in an apply operation");
  if (callee.getType().getNumResults() != 1)
    callee.emitError("expected stencil function to return only one result");
  if (callee.getType().getResult(0) != viewType.getElementType())
    callee.emitError("incompatible stencil function return type and view type");

  result.addOperands(operands);
  result.addAttribute(getCalleeAttrName(), builder->getSymbolRefAttr(callee));
  result.addTypes(viewType);
}

FunctionType stencil::ApplyOp::getCalleeType() {
  SmallVector<Type, 1> resultTypes({getResult()->getType()});
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

static ParseResult parseApplyOp(OpAsmParser &parser, OperationState &state) {
  SymbolRefAttr calleeAttr;
  FunctionType funcType;
  SmallVector<OpAsmParser::OperandType, 3> operands;
  auto calleeLoc = parser.getNameLoc();
  if (parser.parseAttribute(calleeAttr, stencil::ApplyOp::getCalleeAttrName(),
                            state.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType(funcType) ||
      parser.addTypesToList(funcType.getResults(), state.types) ||
      parser.resolveOperands(operands, funcType.getInputs(), calleeLoc,
                             state.operands))
    return failure();

  return success();
}

static void print(stencil::ApplyOp applyOp, OpAsmPrinter &printer) {
  printer << stencil::ApplyOp::getOperationName() << ' '
          << applyOp.getAttr(stencil::ApplyOp::getCalleeAttrName()) << '(';
  printer.printOperands(applyOp.getOperands());
  printer << ')';
  printer.printOptionalAttrDict(applyOp.getAttrs(),
                                {stencil::ApplyOp::getCalleeAttrName()});
  printer << " : ";

  FunctionType calleeType = applyOp.getCalleeType();
  FunctionType funcType = FunctionType::get(
      calleeType.getInputs(), llvm::makeArrayRef({applyOp.res()->getType()}),
      applyOp.getContext());

  printer.printType(funcType);
}

static LogicalResult verify(stencil::ApplyOp applyOp) {
  // Check that the callee attribute was specified.
  auto fnAttr = applyOp.getAttrOfType<SymbolRefAttr>(
      stencil::ApplyOp::getCalleeAttrName());
  if (!fnAttr)
    return applyOp.emitOpError("requires a '")
           << stencil::ApplyOp::getCalleeAttrName()
           << "' symbol reference attribute";
  auto fn = applyOp.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      fnAttr.getLeafReference());
  if (!fn)
    return applyOp.emitOpError() << "'" << fnAttr.getLeafReference()
                                 << "' does not reference a valid function";
  if (!fn.getAttr(stencil::StencilDialect::getStencilFunctionAttrName()))
    return applyOp.emitOpError() << "'" << fnAttr.getLeafReference()
                                 << "' does not reference a stencil function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != applyOp.getNumOperands())
    return applyOp.emitOpError("incorrect number of operands for callee");

  /*for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (applyOp.getOperand(i)->getType() != fnType.getInput(i))
      return applyOp.emitOpError("operand type mismatch");*/

  if (fnType.getNumResults() != 1)
    return applyOp.emitOpError("incorrect number of results for callee");

  Type elementType =
      applyOp.res()->getType().cast<stencil::ViewType>().getElementType();
  if (fnType.getResult(0) != elementType)
    return applyOp.emitOpError("element type mismatch");

  return success();
}

//===----------------------------------------------------------------------===//
// stencil.call
//===----------------------------------------------------------------------===//

void stencil::CallOp::build(Builder *builder, OperationState &result,
                            FuncOp callee, stencil::ViewType viewType,
                            ArrayRef<int64_t> offset,
                            ArrayRef<Value *> operands) {
  assert(offset.size() == 3 && "expected offset with 3 elements");
  assert(
      callee.getAttr(stencil::StencilDialect::getStencilFunctionAttrName()) &&
      "only stencil functions can be used in an apply operation");
  assert(callee.getType().getNumResults() == 1 &&
         "expected stencil function to return only one result");
  assert(callee.getType().getResult(0) == viewType.getElementType() &&
         "incompatible stencil function return type "
         "and view type");

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
  auto fnAttr =
      callOp.getAttrOfType<SymbolRefAttr>(stencil::CallOp::getCalleeAttrName());
  if (!fnAttr)
    return callOp.emitOpError("requires a '")
           << stencil::CallOp::getCalleeAttrName()
           << "' symbol reference attribute";
  auto fn = callOp.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      fnAttr.getLeafReference());
  if (!fn)
    return callOp.emitOpError() << "'" << fnAttr.getLeafReference()
                                << "' does not reference a valid function";
  if (!fn.getAttr(stencil::StencilDialect::getStencilFunctionAttrName()))
    return callOp.emitOpError() << "'" << fnAttr.getLeafReference()
                                << "' does not reference a stencil function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != callOp.getNumOperands())
    return callOp.emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (callOp.getOperand(i)->getType() != fnType.getInput(i))
      return callOp.emitOpError("operand type mismatch");

  if (fnType.getNumResults() != 1)
    return callOp.emitOpError("incorrect number of results for callee");

  return success();
}

namespace mlir {
namespace stencil {
#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.cpp.inc"
} // namespace stencil
} // namespace mlir
