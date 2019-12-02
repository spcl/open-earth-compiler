// Dialect/Sten/StenDialect.h

#ifndef MLIR_DIALECT_STEN_STENDIALECT_H
#define MLIR_DIALECT_STEN_STENDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace sten {

class StenDialect : public Dialect {
public:
  explicit StenDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to Sten operations
  static StringRef getDialectNamespace() { return "sten"; }

  /// Parses a type registered to this dialect
  Type parseType(DialectAsmParser &parser) const override;

  /// Prints a type registered to this dialect
  void printType(Type type, DialectAsmPrinter &os) const override;
};

} // namespace sten
} // namespace mlir

#endif // MLIR_DIALECT_STEN_STENDIALECT_H
