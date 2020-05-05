#ifndef DIALECT_STENCIL_STENCILTYPES_H
#define DIALECT_STENCIL_STENCILTYPES_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

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
  static constexpr int32_t kDynamicSize = -1;
  static constexpr int32_t kScalarDim = 0;

  /// Return the element type
  Type getElementType() const;

  /// Return the shape of the type
  ArrayRef<int> getShape() const;

  /// Support isa, cast, and dyn_cast
  static bool classof(Type type) {
    return type.getKind() == StencilTypes::Field ||
           type.getKind() == StencilTypes::Temp;
  }

  /// Return true if the dimension size is dynamic
  static constexpr bool isDynamic(int32_t dimSize) {
    return dimSize == kDynamicSize;
  }
  /// Return true for scalarized dimensions
  static constexpr bool isScalar(int32_t dimSize) {
    return dimSize == kScalarDim;
  }
};

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

/// Fields are multi-dimensional input and output arrays
class FieldType
    : public Type::TypeBase<FieldType, GridType, detail::FieldTypeStorage> {
public:
  using Base::Base;

  static FieldType get(MLIRContext *context, Type elementType,
                       ArrayRef<int> dimensions);

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
                      ArrayRef<int> dimensions);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == StencilTypes::Temp; }
};

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILTYPES_H
