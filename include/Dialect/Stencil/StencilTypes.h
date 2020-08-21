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
  int64_t getRank() const { return getShape().size(); }

  /// Return true if all dimensions have a dynamic shape
  int64_t hasDynamicShape() const {
    return llvm::all_of(getShape(), [](int64_t size) {
      return size == kDynamicDimension || size == kScalarDimension;
    });
  }

  /// Return true if all dimensions have a static
  int64_t hasStaticShape() const {
    return llvm::none_of(
        getShape(), [](int64_t size) { return size == kDynamicDimension; });
  }

  /// Return the allocated / non-scalar dimensions
  SmallVector<bool, 3> getAllocation() const {
    SmallVector<bool, 3> result;
    result.resize(getRank());
    llvm::transform(getShape(), result.begin(),
                    [](int64_t x) { return x != kScalarDimension; });
    return result;
  }

  /// Return the compatible memref shape
  /// (reverse shape from column-major to row-major)
  SmallVector<int64_t, 3> getMemRefShape() const {
    SmallVector<int64_t, 3> result;
    for (auto size : llvm::reverse(getShape())) {
      switch (size) {
      case (kDynamicDimension):
        result.push_back(ShapedType::kDynamicSize);
        break;
      case (kScalarDimension):
        break;
      default:
        result.push_back(size);
      }
    }
    return result;
  }

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

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILTYPES_H
