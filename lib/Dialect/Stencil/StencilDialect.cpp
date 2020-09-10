#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::stencil;

//===----------------------------------------------------------------------===//
// Stencil Dialect
//===----------------------------------------------------------------------===//

StencilDialect::StencilDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<StencilDialect>()) {
  addTypes<FieldType, TempType, ResultType>();

  addOperations<
#define GET_OP_LIST
#include "Dialect/Stencil/StencilOps.cpp.inc"
      >();

  // Allow Stencil operations to exist in their generic form
  allowUnknownOperations();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

Type StencilDialect::parseType(DialectAsmParser &parser) const {
  StringRef prefix;
  // Parse the prefix
  if (parser.parseKeyword(&prefix)) {
    parser.emitError(parser.getNameLoc(), "expected type identifier");
    return Type();
  }

  // Parse a result type
  if (prefix == getResultTypeName()) {
    // Parse the element type
    Type resultType;
    if (parser.parseLess() || parser.parseType(resultType) ||
        parser.parseGreater()) {
      parser.emitError(parser.getNameLoc(), "expected valid result type");
      return Type();
    }
    return ResultType::get(resultType);
  }

  // Parse a field or temp type
  if (prefix == getFieldTypeName() || prefix == getTempTypeName()) {
    // Parse the shape
    SmallVector<int64_t, 3> shape;
    if (parser.parseLess() || parser.parseDimensionList(shape)) {
      parser.emitError(parser.getNameLoc(), "expected valid dimension list");
      return Type();
    }

    // Parse the element type
    Type elementType;
    if (parser.parseType(elementType) || parser.parseGreater()) {
      parser.emitError(parser.getNameLoc(), "expected valid element type");
      return Type();
    }

    // Return the Stencil type
    if (prefix == getFieldTypeName())
      return FieldType::get(elementType, shape);
    else
      return TempType::get(elementType, shape);
  }

  // Failed to parse a stencil type
  parser.emitError(parser.getNameLoc(), "unknown stencil type ")
      << parser.getFullSymbolSpec();
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

namespace {

void printGridType(StringRef name, Type type, DialectAsmPrinter &printer) {
  printer << name;
  printer << "<";
  for (auto size : type.cast<GridType>().getShape()) {
    if (size == GridType::kDynamicDimension)
      printer << "?";
    else
      printer << size;
    printer << "x";
  }
  printer << type.cast<GridType>().getElementType() << ">";
}

void printResultType(StringRef name, Type type, DialectAsmPrinter &printer) {
  printer << name;
  printer << "<" << type.cast<ResultType>().getResultType() << ">";
}

} // namespace

void StencilDialect::printType(Type type, DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
      .Case<FieldType>(
          [&](Type) { printGridType(getFieldTypeName(), type, printer); })
      .Case<TempType>(
          [&](Type) { printGridType(getTempTypeName(), type, printer); })
      .Case<ResultType>(
          [&](Type) { printResultType(getResultTypeName(), type, printer); })
      .Default([](Type) { llvm_unreachable("unexpected 'shape' type kind"); });
}
