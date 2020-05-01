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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <bits/stdint-intn.h>
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
  // Get the type prefix
  StringRef prefix;
  parser.parseKeyword(&prefix);

  // Helper method converting a string to a dimension vector
  auto parseDimensions = [&](StringRef input) {
    std::vector<int> result;
    for (size_t i = 0, e = input.size(); i != e; ++i) {
      switch (input[i]) {
      case 'i':
        result.push_back(kIDimension);
        break;
      case 'j':
        result.push_back(kJDimension);
        break;
      case 'k':
        result.push_back(kKDimension);
        break;
      default:
        parser.emitError(parser.getNameLoc(),
                         "unexpected dimension identifier");
        return llvm::Optional<std::vector<int>>();
      }
    }
    // Verify the dimension vector
    if (!std::is_sorted(result.begin(), result.end())) {
      parser.emitError(parser.getNameLoc(), "expected sorted dimension list");
      return llvm::Optional<std::vector<int>>();
    }
    if (std::unique(result.begin(), result.end()) != result.end()) {
      parser.emitError(parser.getNameLoc(), "expected unique dimension list");
      return llvm::Optional<std::vector<int>>();
    }
    return llvm::Optional<std::vector<int>>(result);
  };

  // Parse a field type
  if (prefix == getFieldTypeName()) {
    StringRef identifiers;
    Type elementType;
    if (parser.parseLess() || parser.parseKeyword(&identifiers) ||
        parser.parseComma() || parser.parseType(elementType) ||
        parser.parseGreater()) {
      return Type();
    }
    auto dimensions = parseDimensions(identifiers);
    if (!dimensions.hasValue())
      return Type();
    return FieldType::get(getContext(), elementType, dimensions.getValue());
  }

  // Parse a temp type
  else if (prefix == getTempTypeName()) {
    StringRef identifiers;
    Type elementType;
    if (parser.parseLess() || parser.parseKeyword(&identifiers) ||
        parser.parseComma() || parser.parseType(elementType) ||
        parser.parseGreater()) {
      return Type();
    }
    auto dimensions = parseDimensions(identifiers);
    if (!dimensions.hasValue())
      return Type();
    return TempType::get(getContext(), elementType, dimensions.getValue());
  }

  parser.emitError(parser.getNameLoc(), "unknown stencil type: ")
      << parser.getFullSymbolSpec();
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

namespace {

StringRef dimensionToString(int dimension) {
  StringRef result = "";
  switch (dimension) {
  case kIDimension:
    result = "i";
    break;
  case kJDimension:
    result = "j";
    break;
  case kKDimension:
    result = "k";
    break;
  default:
    llvm_unreachable("dimension not supported");
  }
  return result;
}

void print(FieldType fieldType, DialectAsmPrinter &printer) {
  printer << StencilDialect::getFieldTypeName() << "<";
  ArrayRef<int> dimensions = fieldType.getDimensions();
  for (auto dimension : dimensions)
    printer << dimensionToString(dimension);
  printer << ",";
  printer << fieldType.getElementType() << ">";
}

void print(TempType tempType, DialectAsmPrinter &printer) {
  printer << StencilDialect::getTempTypeName() << "<";
  ArrayRef<int> dimensions = tempType.getDimensions();
  for (auto dimension : dimensions)
    printer << dimensionToString(dimension);
  printer << ",";
  printer << tempType.getElementType() << ">";
}

} // namespace

void StencilDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  case StencilTypes::Field:
    print(type.cast<FieldType>(), printer);
    break;
  case StencilTypes::Temp:
    print(type.cast<TempType>(), printer);
    break;
  default:
    llvm_unreachable("unhandled stencil type");
  }
}
