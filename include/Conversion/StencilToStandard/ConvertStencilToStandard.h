#ifndef CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H
#define CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H

#include "Dialect/Stencil/StencilOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include <bits/stdint-intn.h>
#include <tuple>

namespace mlir {
namespace stencil {

/// Convert stencil types to standard types
struct StencilTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  /// Create a stencil type converter using the default conversions
  StencilTypeConverter(MLIRContext *context);

  /// Convert a dynamic field to a statically sized memref
  Type convertFieldType(FieldType fieldType, ArrayRef<int64_t> shape);

  /// Return the context
  MLIRContext *getContext() { return context; }

private:
  MLIRContext *context;
};

/// Base class for the stencil to standard operation conversions
class StencilToStdPattern : public ConversionPattern {
public:
  StencilToStdPattern(StringRef rootOpName, StencilTypeConverter &typeConverter,
                      PatternBenefit benefit = 1);

  /// Compute the shape of the operation
  Index computeShape(ShapeOp shapeOp) const;

  /// Compute offset, shape, strides of the subview
  std::tuple<Index, Index, Index> computeSubViewShape(FieldType fieldType,
                                                      ShapeOp accessOp,
                                                      ShapeOp assertOp) const;

  /// Compute the index values for a given constant offset
  SmallVector<Value, 3>
  computeIndexValues(ValueRange inductionVars, Index offset,
                     ArrayRef<bool> allocation,
                     ConversionPatternRewriter &rewriter) const;

  /// Return operation of a specific type that uses a given value
  template <typename OpTy>
  OpTy getUserOp(Value value) const {
    for (auto user : value.getUsers())
      if (OpTy op = dyn_cast<OpTy>(user))
        return op;
    return nullptr;
  }

protected:
  /// Reference to the type converter
  StencilTypeConverter &typeConverter;
};

/// Helper class to implement patterns that match one source operation
template <typename OpTy>
class StencilOpToStdPattern : public StencilToStdPattern {
public:
  StencilOpToStdPattern(StencilTypeConverter &typeConverter,
                        PatternBenefit benefit = 1)
      : StencilToStdPattern(OpTy::getOperationName(), typeConverter, benefit) {}
};

/// Helper method to populate the conversion pattern list
void populateStencilToStdConversionPatterns(StencilTypeConverter &typeConveter,
                                            OwningRewritePatternList &patterns);

} // namespace stencil
} // namespace mlir

#endif // CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD_H
