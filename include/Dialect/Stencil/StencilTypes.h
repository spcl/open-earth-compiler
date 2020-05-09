#ifndef DIALECT_STENCIL_STENCILTYPES_H
#define DIALECT_STENCIL_STENCILTYPES_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <bits/stdint-intn.h>

namespace mlir {
namespace stencil {

namespace detail {
struct GridTypeStorage;
struct FieldTypeStorage;
struct TempTypeStorage;
} // namespace detail

namespace StencilTypes {
enum Kind {
  Field = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  Temp,
  LAST_USED_PRIVATE_EXPERIMENTAL_0_TYPE = Temp
};
}

//===----------------------------------------------------------------------===//
// GridType
//===----------------------------------------------------------------------===//

/// Base class of the field and view types.
class GridType : public Type {
public:
  using ImplType = detail::GridTypeStorage;
  using Type::Type;

  /// Constants used to mark dynamic size or scalarized dimensions
  static constexpr int64_t kDynamicDimension = -1;
  static constexpr int64_t kScalarDimension = 0;

  /// Return the element type
  Type getElementType() const;

  /// Return the shape of the type
  ArrayRef<int64_t> getShape() const;

  /// Return the rank of the type
  int64_t getRank() const { return getShape().size(); }

  /// Return true if all dimensions are dynamic
  int64_t hasStaticShape() const {
    return llvm::none_of(getShape(), [](int64_t size) {
      return size == kDynamicDimension;
    });
  }

  /// Return the allocated / non-scalar dimensions
  SmallVector<bool, 3> getAllocation() const {
    SmallVector<bool, 3> result;
    result.resize(getRank());
    llvm::transform(getShape(), result.begin(),
                    [](int64_t x) { return x != kScalarDimension; });
    return result;
  }

  /// Support isa, cast, and dyn_cast
  static bool classof(Type type) {
    return type.getKind() == StencilTypes::Field ||
           type.getKind() == StencilTypes::Temp;
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

  static FieldType get(MLIRContext *context, Type elementType,
                       ArrayRef<int64_t> shape);

  /// Used to implement LLVM-style casts
  static bool kindof(unsigned kind) { return kind == StencilTypes::Field; }
};

//===----------------------------------------------------------------------===//
// TempType
//===----------------------------------------------------------------------===//

/// Temporaries keep multi-dimensional intermediate results
class TempType
    : public Type::TypeBase<TempType, GridType, detail::TempTypeStorage> {
public:
  using Base::Base;

  static TempType get(MLIRContext *context, Type elementType,
                      ArrayRef<int64_t> shape);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == StencilTypes::Temp; }
};

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILTYPES_H
