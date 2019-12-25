#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"

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

  // Helper method converting a string to an allocation
  auto parseDimensions = [&](StringRef input) -> std::vector<int> {
    std::vector<int> result;
    for (int i = 0, e = input.size(); i != e; ++i) {
      switch (input[i]) {
      case 'i':
        result.push_back(0);
        break;
      case 'j':
        result.push_back(1);
        break;
      case 'k':
        result.push_back(2);
        break;
      default:
        parser.emitError(parser.getNameLoc(),
                         "unexpected dimension identifier");
      }
    }
    return result;
  };

  // Parse a field type
  if (prefix == getFieldTypeName()) {
    StringRef dimensions;
    Type elementType;
    if (parser.parseLess() || parser.parseKeyword(&dimensions) ||
        parser.parseComma() || parser.parseType(elementType) ||
        parser.parseGreater()) {
      return Type();
    }
    return FieldType::get(getContext(), elementType,
                          parseDimensions(dimensions));
  }
  // Parse a view type
  else if (prefix == getViewTypeName()) {
    StringRef dimensions;
    Type elementType;
    if (parser.parseLess() || parser.parseKeyword(&dimensions) ||
        parser.parseComma() || parser.parseType(elementType) ||
        parser.parseGreater()) {
      return Type();
    }
    return ViewType::get(getContext(), elementType,
                         parseDimensions(dimensions));
  }

  parser.emitError(parser.getNameLoc(), "unknown Stencil type: ")
      << parser.getFullSymbolSpec();
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

namespace {

StringRef dimensionToString(int dimension) {
  switch (dimension) {
  case 0:
    return "i";
  case 1:
    return "j";
  case 2:
    return "k";
  default:
    assert(false && "dimension not supported");
  }
  return "";
}

void print(FieldType fieldType, DialectAsmPrinter &printer) {
  printer << StencilDialect::getFieldTypeName() << "<";
  ArrayRef<int> dimensions = fieldType.getDimensions();
  for (auto dimension : dimensions)
    printer << dimensionToString(dimension);
  printer << ",";
  printer << fieldType.getElementType() << ">";
}

void print(ViewType viewType, DialectAsmPrinter &printer) {
  printer << StencilDialect::getViewTypeName() << "<";
  ArrayRef<int> dimensions = viewType.getDimensions();
  for (auto dimension : dimensions)
    printer << dimensionToString(dimension);
  printer << ",";
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
