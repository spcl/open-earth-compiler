#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <string>

using namespace mlir;
using namespace mlir::stencil;

//===----------------------------------------------------------------------===//
// Stencil Dialect
//===----------------------------------------------------------------------===//

StencilDialect::StencilDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<FieldType, TempType>();

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
  
  // Parse the shape
  SmallVector<int64_t, 3> shape;
  if(parser.parseLess() || parser.parseDimensionList(shape)) {
    parser.emitError(parser.getNameLoc(), "expected valid dimension list");
    return Type();
  }

  // Parse the element type
  Type elementType;
  if(parser.parseType(elementType) || parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected valid element type");
    return Type();
  }

  // Return the Stencil type
  if (prefix == getFieldTypeName())
    return FieldType::get(getContext(), elementType, shape);
  if (prefix == getTempTypeName())
    return TempType::get(getContext(), elementType, shape);

  // Failed to parse a stencil type
  parser.emitError(parser.getNameLoc(), "unknown stencil type ")
      << parser.getFullSymbolSpec();
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

namespace {

void print(GridType gridType, DialectAsmPrinter &printer) {
  printer << "<";
  for (auto size : gridType.getShape()) {
    if(size == GridType::kDynamicDimension)
      printer << "?";
    else 
      printer << size;
    printer << "x";
  }
  printer << gridType.getElementType() << ">";
}

} // namespace

void StencilDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  case StencilTypes::Field:
    printer << StencilDialect::getFieldTypeName();
    print(type.cast<GridType>(), printer);
    break;
  case StencilTypes::Temp:
    printer << StencilDialect::getTempTypeName();
    print(type.cast<GridType>(), printer);
    break;
  default:
    llvm_unreachable("unhandled stencil type");
  }
}
