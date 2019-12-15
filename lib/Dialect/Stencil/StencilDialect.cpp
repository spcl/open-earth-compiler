#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"

using namespace mlir;
using namespace mlir::stencil;

//===----------------------------------------------------------------------===//
// Stencil Dialect
//===----------------------------------------------------------------------===//

StencilDialect::StencilDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<FieldType, ViewType>();

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
  // Get the type prefix
  StringRef prefix;
  parser.parseKeyword(&prefix);

  // Parse a field type
  if (prefix == "field") {
    SmallVector<int64_t, 3> shape;
    Type elementType;
    if (parser.parseLess() || parser.parseDimensionList(shape) ||
        parser.parseType(elementType) || parser.parseGreater()) {
      return Type();
    }
    // Make sure that we are only dealing with 3D fields
    if (shape.size() != 3)
      parser.emitError(parser.getNameLoc(),
                       "expected field to have three dimensions");
    return FieldType::get(getContext(), elementType, shape);
  }
  // Parse a view type
  else if (prefix == "view") {
    SmallVector<int64_t, 3> shape;
    Type elementType;
    if (parser.parseLess() || parser.parseDimensionList(shape) ||
        parser.parseType(elementType) || parser.parseGreater()) {
      return Type();
    }
    // Make sure that we are only dealing with 3D views
    if (shape.size() != 3)
      parser.emitError(parser.getNameLoc(),
                       "expected view to have three dimensions");
    return ViewType::get(getContext(), elementType, shape);
  }

  parser.emitError(parser.getNameLoc(), "unknown Stencil type: ")
      << parser.getFullSymbolSpec();
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

namespace {

void print(FieldType fieldType, DialectAsmPrinter &printer) {
  printer << "field<";
  for (auto &elem : fieldType.getShape()) {
    if (elem < 0)
      printer << "?";
    else
      printer << elem;
    printer << "x";
  }
  printer << fieldType.getElementType() << ">";
}

void print(ViewType viewType, DialectAsmPrinter &printer) {
  printer << "view<";
  for (auto &elem : viewType.getShape()) {
    if (elem < 0)
      printer << "?";
    else
      printer << elem;
    printer << "x";
  }
  printer << viewType.getElementType() << ">";
}

} // namespace

void StencilDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  case StencilTypes::Field:
    print(type.cast<FieldType>(), printer);
    break;
  case StencilTypes::View:
    print(type.cast<ViewType>(), printer);
    break;
  default:
    llvm_unreachable("unhandled Stencil type");
  }
}
