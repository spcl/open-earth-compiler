#ifndef DIALECT_STENCIL_STENCILTYPES_H
#define DIALECT_STENCIL_STENCILTYPES_H

#include "Dialect/Stencil/StencilDialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

namespace mlir {
namespace stencil {

namespace detail {
struct GridTypeStorage;
struct FieldTypeStorage;
struct TempTypeStorage;
struct ResultTypeStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// GridType
//===----------------------------------------------------------------------===//

/// Base class of the field and view types.
class GridType : public Type {
public:
  using ImplType = detail::GridTypeStorage;
  using Type::Type;

  static bool classof(Type type);

  /// Constants used to mark dynamic size or scalarized dimensions
  static constexpr int64_t kDynamicDimension = -1;
  static constexpr int64_t kScalarDimension = 0;

  /// Return the element type
  Type getElementType() const;

  /// Return the shape of the type
  ArrayRef<int64_t> getShape() const;

  /// Return the rank of the type
  int64_t getRank() const;

  /// Return true if all dimensions have a dynamic shape
  int64_t hasDynamicShape() const;

  /// Return true if all dimensions have a static
  int64_t hasStaticShape() const;

  /// Return the allocated / non-scalar dimensions
  SmallVector<bool, 3> getAllocation() const;

  /// Return the compatible memref shape
  /// (reverse shape from column-major to row-major)
  SmallVector<int64_t, 3> getMemRefShape() const;

  /// Return true if the dimension size is dynamic
  static constexpr bool isDynamic(int64_t dimSize) {
    return dimSize == kDynamicDimension;
  }

  /// Return true for scalarized dimensions
  static constexpr bool isScalar(int64_t dimSize) {
    return dimSize == kScalarDimension;
  }
}; // namespace stencil

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

/// Fields are multi-dimensional input and output arrays
class FieldType
    : public Type::TypeBase<FieldType, GridType, detail::FieldTypeStorage> {
public:
  using Base::Base;

  static FieldType get(Type elementType, ArrayRef<int64_t> shape);
};

//===----------------------------------------------------------------------===//
// TempType
//===----------------------------------------------------------------------===//

/// Temporaries keep multi-dimensional intermediate results
class TempType
    : public Type::TypeBase<TempType, GridType, detail::TempTypeStorage> {
public:
  using Base::Base;

  static TempType get(Type elementType, ArrayRef<int64_t> shape);
};

//===----------------------------------------------------------------------===//
// ResultType
//===----------------------------------------------------------------------===//

/// Temporaries keep multi-dimensional intermediate results
class ResultType
    : public Type::TypeBase<ResultType, Type, detail::ResultTypeStorage> {
public:
  using Base::Base;

  static ResultType get(Type resultType);

  /// Return the result type
  Type getResultType() const;
};

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILTYPES_H
