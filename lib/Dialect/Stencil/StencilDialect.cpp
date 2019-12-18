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
  auto parseAllocation = [&](StringRef Input) {
    if (Input == "ijk")
      return StencilStorage::IJK;
    else if (Input == "ij")
      return StencilStorage::IJ;
    else if (Input == "ik")
      return StencilStorage::IK;
    else if (Input == "jk")
      return StencilStorage::JK;
    else if (Input == "i")
      return StencilStorage::I;
    else if (Input == "j")
      return StencilStorage::J;
    else if (Input == "k")
      return StencilStorage::K;
    parser.emitError(parser.getNameLoc(), "unexpected field allocation type");
    return StencilStorage::IJK;
  };

  // Parse a field type
  if (prefix == getFieldTypeName()) {
    StringRef allocation;
    Type elementType;
    if (parser.parseLess() || parser.parseKeyword(&allocation) ||
        parser.parseComma() || parser.parseType(elementType) ||
        parser.parseGreater()) {
      return Type();
    }
    return FieldType::get(getContext(), elementType,
                          parseAllocation(allocation));
  }
  // Parse a view type
  else if (prefix == getViewTypeName()) {
    StringRef allocation;
    Type elementType;
    if (parser.parseLess() || parser.parseKeyword(&allocation) ||
        parser.parseComma() || parser.parseType(elementType) ||
        parser.parseGreater()) {
      return Type();
    }
    return ViewType::get(getContext(), elementType,
                         parseAllocation(allocation));
  }

  parser.emitError(parser.getNameLoc(), "unknown Stencil type: ")
      << parser.getFullSymbolSpec();
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

namespace {

StringRef printAllocation(StencilStorage::Allocation allocation) {
  switch(allocation) {
    case mlir::stencil::StencilStorage::IJK:
      return "ijk";
    case mlir::stencil::StencilStorage::IJ:
      return "ij";
    case mlir::stencil::StencilStorage::IK:
      return "ik";
    case mlir::stencil::StencilStorage::JK:
      return "jk";
    case mlir::stencil::StencilStorage::I:
      return "i";
    case mlir::stencil::StencilStorage::J:
      return "j";
    case mlir::stencil::StencilStorage::K:
      return "k";
    default:
      assert(false && "unexpected allocation type");
  }
  return "ijk";
}

void print(FieldType fieldType, DialectAsmPrinter &printer) {
  printer << StencilDialect::getFieldTypeName() << "<";
  StencilStorage::Allocation allocation = fieldType.getAllocation();
  printer << printAllocation(allocation);
  printer << ",";
  printer << fieldType.getElementType() << ">";
}

void print(ViewType viewType, DialectAsmPrinter &printer) {
  printer << StencilDialect::getViewTypeName() << "<";
  StencilStorage::Allocation allocation = viewType.getAllocation();
  printer << printAllocation(allocation);
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
