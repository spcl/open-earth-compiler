#ifndef CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H
#define CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H

#include "Dialect/Stencil/StencilOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace stencil {

/// Convert stencil types to standard types
struct StencilTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  /// Create a stencil type converter using the default conversions
  StencilTypeConverter();

  /// Convert stencil field types to memref types
  Type convertFieldType(FieldType type);
};

/// Base class for the stencil to standard operation conversions
class StencilToStdPattern : public ConversionPattern {
public:
  StencilToStdPattern(StringRef rootOpName, MLIRContext *context,
                           StencilTypeConverter &typeConverter,
                           PatternBenefit benefit = 1);

protected:
  /// Reference to the type converter
  StencilTypeConverter &typeConverter;
};

/// Helper class to implement patterns that match one source operation
template <typename OpTy>
class StencilOpToStdPattern : public StencilToStdPattern {
public:
  StencilOpToStdPattern(MLIRContext *context,
                             StencilTypeConverter &typeConverter,
                             PatternBenefit benefit = 1)
      : StencilToStdPattern(OpTy::getOperationName(), context,
                                 typeConverter, benefit) {}
};

/// Helper method to populate the conversion pattern list
void populateStencilToStdConversionPatterns(
    MLIRContext *ctx, StencilTypeConverter &typeConveter,
    OwningRewritePatternList &patterns);

} // namespace stencil
} // namespace mlir

#endif // CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H
