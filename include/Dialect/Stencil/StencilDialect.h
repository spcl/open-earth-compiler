#ifndef DIALECT_STENCIL_STENCILDIALECT_H
#define DIALECT_STENCIL_STENCILDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include <cstdint>

namespace mlir {
namespace stencil {

// Constant dimension identifiers
constexpr static int kIDimension = 0;
constexpr static int kJDimension = 1;
constexpr static int kKDimension = 2;

// Index type size
constexpr static int64_t kIndexSize = 3;

// Index type used to store offsets and bounds
typedef SmallVector<int64_t, kIndexSize> Index;

class StencilDialect : public Dialect {
public:
  explicit StencilDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to stencil operations
  static StringRef getDialectNamespace() { return "stencil"; }

  static StringRef getStencilFunctionAttrName() { return "stencil.function"; }
  static StringRef getStencilProgramAttrName() { return "stencil.program"; }

  static StringRef getFieldTypeName() { return "field"; }
  static StringRef getTempTypeName() { return "temp"; }

  static bool isStencilFunction(FuncOp funcOp) {
    return !!funcOp.getAttr(getStencilFunctionAttrName());
  }
  static bool isStencilProgram(FuncOp funcOp) {
    return !!funcOp.getAttr(getStencilProgramAttrName());
  }

  /// Parses a type registered to this dialect
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(Type type, DialectAsmPrinter &os) const override;
};

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILDIALECT_H
