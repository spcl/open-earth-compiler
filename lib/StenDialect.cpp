// Dialect/Sten/StenDialect.cpp

#include "sten/StenDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "sten/StenOps.h"
#include "sten/StenTypes.h"

using namespace mlir;
using namespace mlir::sten;

//===----------------------------------------------------------------------===//
// Sten Dialect
//===----------------------------------------------------------------------===//

StenDialect::StenDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  // Allow Sten operations to exist in their generic form
  allowUnknownOperations();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

Type StenDialect::parseType(DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown Sten type: ")
      << parser.getFullSymbolSpec();
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void StenDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("unhandled Sten type");
  }
}
